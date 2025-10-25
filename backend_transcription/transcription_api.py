"""
FastAPI Transcription Service with GPU/MIG Pooling
Dedicated service for audio transcription using GPU resource pooling.
"""

import os
import uuid
import asyncio
import logging
import tempfile
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch

from gpu_pool import GPUPool, get_gpu_pool, initialize_gpu_pool
import whisper
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Transcription API",
    description="FastAPI service for audio transcription with GPU pooling",
    version="1.0.0"
)

# Global variables
whisper_models = {}  # Dictionary to store models per GPU slice
whisper_processors = {}  # Dictionary to store processors per GPU slice
indic_models = {}  # Dictionary to store IndicConformer models per GPU slice
gpu_pool = None

class TranscriptionRequest(BaseModel):
    """Request model for transcription"""
    request_id: str
    slice_id: str
    audio_path: str
    language_code: Optional[str] = None

class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    success: bool
    request_id: str
    transcription: Optional[str] = None
    detected_language: Optional[str] = None
    language_code: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class TranscriptionService:
    """Service for handling transcription requests with GPU pooling"""
    
    def __init__(self):
        self.active_requests: Dict[str, TranscriptionRequest] = {}
        self.results: Dict[str, TranscriptionResponse] = {}
    
    def preprocess_audio(self, audio_path: str, target_sr: int = None) -> tuple:
        """
        Preprocess audio file: convert to mono and resample to target sampling rate.
        EXACT copy from old backend for maximum compatibility.
        """
        if target_sr is None:
            target_sr = 16000
            
        # Try to fix recorded audio issues first
        fixed_audio_path = self.fix_recorded_audio_issues(audio_path)
        use_fixed_file = fixed_audio_path != audio_path
        
        try:
            # Try to load audio file with different backends
            waveform, orig_sr = None, None
            
            # Use fixed file if available
            file_to_load = fixed_audio_path if use_fixed_file else audio_path
            
            # List of backends to try in order of preference
            backends_to_try = ['ffmpeg', 'sox_io', 'soundfile']
            
            for backend in backends_to_try:
                try:
                    logger.debug(f"Trying to load audio with {backend} backend")
                    import torchaudio
                    waveform, orig_sr = torchaudio.load(file_to_load, backend=backend)
                    logger.debug(f"Successfully loaded audio with {backend} backend")
                    break
                except Exception as backend_error:
                    logger.debug(f"Failed to load with {backend}: {backend_error}")
                    continue
            
            # If all backends failed, try without specifying backend
            if waveform is None:
                try:
                    logger.debug("Trying to load audio with default backend")
                    import torchaudio
                    waveform, orig_sr = torchaudio.load(file_to_load)
                    logger.debug("Successfully loaded audio with default backend")
                except Exception as default_error:
                    logger.debug(f"Default backend failed: {default_error}")
                    
                    # Final fallback: use librosa with different parameters for recorded audio
                    try:
                        logger.debug("Trying to load audio with librosa")
                        audio_data = None
                        orig_sr = None
                        clean_wav = self.ensure_wav(file_to_load)
                        audio_data, orig_sr = librosa.load(clean_wav, sr=None, mono=True)

                        # Try multiple strategies for recorded audio files
                        strategies = [
                            # Strategy 1: Default librosa (mono=True)
                            lambda: librosa.load(file_to_load, sr=None, mono=True),
                            # Strategy 2: Keep stereo, then convert to mono
                            lambda: librosa.load(file_to_load, sr=None, mono=False),
                            # Strategy 3: Force specific sample rate
                            lambda: librosa.load(file_to_load, sr=16000, mono=True),
                            # Strategy 4: Try with different resampling
                            lambda: librosa.load(file_to_load, sr=None, mono=True, res_type='kaiser_fast'),
                            # Strategy 5: Try with offset and duration
                            lambda: librosa.load(file_to_load, sr=None, mono=True, offset=0.0, duration=None),
                        ]
                        
                        audio_data, orig_sr = None, None

                        for i, strategy in enumerate(strategies, start=1):
                            try:
                                logger.debug(f"Trying librosa strategy {i}")
                                audio_data, orig_sr = strategy()
                                logger.debug(f"Strategy {i} succeeded: shape={audio_data.shape}, sr={orig_sr}")
                                break
                            except Exception as e:
                                logger.warning(f"Strategy {i} failed: {e}", exc_info=True)
                                continue
                        
                        # If all strategies failed, try audioread as last resort
                        if audio_data is None:
                            logger.debug("All librosa strategies failed, trying audioread")
                            import audioread
                            with audioread.audio_open(file_to_load) as f:
                                audio_data = []
                                for frame in f:
                                    audio_data.extend(frame)
                                audio_data = np.array(audio_data, dtype=np.float32)
                                orig_sr = f.samplerate
                                logger.debug(f"Audioread succeeded: shape={audio_data.shape}, sr={orig_sr}")
                        
                        # Ensure audio_data is 1D (mono)
                        if len(audio_data.shape) > 1:
                            logger.debug(f"Converting from {audio_data.shape} to mono")
                            if audio_data.shape[0] == 2:  # Stereo
                                audio_data = librosa.to_mono(audio_data)
                            else:  # Multi-channel
                                audio_data = np.mean(audio_data, axis=0)
                        
                        # Ensure audio_data is float32 and normalized
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32)
                        
                        # Normalize audio to prevent clipping
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 0:
                            audio_data = audio_data / max_val * 0.95
                        
                        # Convert to torch tensor and add channel dimension
                        import torch
                        waveform = torch.from_numpy(audio_data).unsqueeze(0)
                        logger.debug(f"Successfully loaded audio with librosa: final shape={waveform.shape}")
                        
                    except Exception as librosa_error:
                        raise Exception(f"Failed to load audio file with any method. TorchAudio error: {default_error}, Librosa error: {str(librosa_error)}")
            
            # Convert to mono if stereo/multi-channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.debug(f"Converted from {waveform.shape[0]} channels to mono")
                
            # Resample to target sampling rate if needed
            if orig_sr != target_sr:
                import torchaudio
                waveform = torchaudio.functional.resample(
                    waveform,
                    orig_freq=orig_sr,
                    new_freq=target_sr
                )
                logger.debug(f"Resampled from {orig_sr} Hz to {target_sr} Hz")
                
            # Return waveform and sample rate (EXACT same as backend_2.0)
            return waveform, target_sr
            
        except Exception as e:
            raise Exception(f"Error in audio preprocessing: {str(e)}")
        finally:
            # Clean up fixed file if it was created
            if use_fixed_file and os.path.exists(fixed_audio_path):
                try:
                    os.remove(fixed_audio_path)
                    logger.debug(f"Cleaned up fixed audio file: {fixed_audio_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up fixed audio file: {cleanup_error}")

    def ensure_wav(self, file_to_load: str) -> str:
        """Convert audio file to WAV format using FFmpeg"""
        import subprocess
        import tempfile
        from pathlib import Path
        
        tmp_wav = Path(tempfile.mkstemp(suffix=".wav")[1])
        cmd = ["ffmpeg", "-y", "-i", file_to_load, "-ar", "16000", "-ac", "1", str(tmp_wav)]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return str(tmp_wav)

    def transcribe_with_indic_conformer(self, waveform, sr: int, language_code: str, slice_id: str) -> str:
        """Transcribe audio using IndicConformer for Indian languages"""
        try:
            # Get the IndicConformer model for this slice
            indic_model = indic_models[slice_id]
            
            # Use IndicConformer for transcription (EXACT same as backend_2.0)
            with torch.no_grad():
                # Get device from slice_id - all MIG devices are on GPU 1, so use cuda:1
                device = gpu_pool.get_device_for_slice(slice_id)
                waveform_device = waveform.to(device)
                
                # IndicConformer expects specific format - let's match backend_2.0 exactly
                transcription = indic_model(
                    waveform_device, 
                    language_code, 
                    "rnnt"
                )
            
            return transcription.strip() if transcription else ""
            
        except Exception as e:
            logger.error(f"Error in IndicConformer transcription: {str(e)}", exc_info=True)
            # Fallback to Whisper if IndicConformer fails
            logger.warning(f"Falling back to Whisper for request on slice {slice_id}")
            return self.transcribe_with_whisper_processor(waveform, sr, None, slice_id)

    def detect_language_simple(self, audio_path: str) -> str:
        """Simple language detection - you can enhance this"""
        # For now, return 'hi' (Hindi) as default for non-English
        # You can integrate with the main backend's language detection later
        return 'hi'

    def transcribe_with_whisper_processor(self, waveform, sr: int, language_code: Optional[str], slice_id: str) -> str:
        """Transcribe audio using Whisper processor approach (EXACT backend_2.0 method)"""
        try:
            # Get the processor and model for this slice
            whisper_processor = whisper_processors[slice_id]
            whisper_model = whisper_models[slice_id]
            
            # Prepare input features (EXACT same as backend_2.0)
            input_features = whisper_processor(
                waveform.squeeze(),
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features.to(whisper_model.device)
            
            # Generate tokens (EXACT same as backend_2.0)
            predicted_ids = whisper_model.generate(input_features)
            
            # Decode tokens to text (EXACT same as backend_2.0)
            transcription = whisper_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error in Whisper processor transcription: {str(e)}")
            raise

    def fix_recorded_audio_issues(self, audio_path: str) -> str:
        """
        Fix common issues with recorded audio files.
        Returns the path to the fixed audio file (might be the same file).
        """
        try:
            import soundfile as sf
            
            # Try to load the audio
            try:
                y, sr = librosa.load(audio_path, sr=None, mono=False)
            except Exception as e:
                logger.debug(f"Could not load audio for fixing: {e}")
                return audio_path
            
            # Create a temporary fixed file
            base_name = os.path.splitext(audio_path)[0]
            fixed_path = f"{base_name}_fixed.wav"
            
            # Ensure mono
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
                logger.debug("Converted stereo to mono")
            
            # Ensure proper sample rate (16kHz is good for ASR)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
                logger.debug(f"Resampled to {sr}Hz")
            
            # Normalize audio
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.95
                logger.debug("Normalized audio")
            
            # Save as standard WAV format
            sf.write(fixed_path, y, sr, format='WAV', subtype='PCM_16')
            logger.debug(f"Saved fixed audio to: {fixed_path}")
            
            return fixed_path
            
        except Exception as e:
            logger.error(f"Error fixing recorded audio: {e}")
            return audio_path

    async def process_transcription(self, request_id: str, audio_path: str, language_code: Optional[str] = None) -> TranscriptionResponse:
        """
        Process transcription request with GPU pooling using only Whisper.
        
        Args:
            request_id: Unique request identifier
            audio_path: Path to audio file
            language_code: Optional language code for transcription
            
        Returns:
            TranscriptionResponse with results
        """
        start_time = asyncio.get_event_loop().time()
        slice_id = None
        
        try:
            # Get a free GPU slice
            slice_id = await gpu_pool.get_free_slice(request_id, timeout=60)
            if not slice_id:
                return TranscriptionResponse(
                    success=False,
                    request_id=request_id,
                    error="No free GPU slice available"
                )
            
            # Set CUDA device for this request
            device = gpu_pool.get_device_for_slice(slice_id)
            torch.cuda.set_device(device)
            
            logger.info(f"Processing transcription {request_id} on GPU slice {slice_id} (device: {device})")
            
            # Check if models are loaded for this GPU slice
            if slice_id not in whisper_models or slice_id not in indic_models:
                logger.error(f"Models not loaded for slice {slice_id}")
                return TranscriptionResponse(
                    success=False,
                    request_id=request_id,
                    error=f"Models not available for slice {slice_id}"
                )
            
            # Preprocess audio (EXACT same as backend_2.0)
            waveform, sr = self.preprocess_audio(audio_path)
            
            # Improved transcription logic: Use IndicConformer for Indian languages, Whisper for English
            logger.info(f"🔍 DEBUG: Received language_code='{language_code}', type={type(language_code)}")
            if language_code:
                detected_language = language_code
                logger.info(f"🔍 DEBUG: Using provided language_code='{language_code}'")
            else:
                # Simple language detection (can be enhanced later)
                logger.info(f"🔍 DEBUG: No language_code provided, using auto-detection")
                detected_language = self.detect_language_simple(audio_path)
            
            # Choose appropriate model based on language
            logger.info(f"🔍 DEBUG: detected_language='{detected_language}', type={type(detected_language)}")
            if detected_language == "en" or detected_language == "eng":
                # Use Whisper for English
                logger.info(f"🎤 Using Whisper for English transcription (detected_language='{detected_language}')")
                transcription = self.transcribe_with_whisper_processor(waveform, sr, detected_language, slice_id)
            else:
                # Use IndicConformer for Indian languages (much better quality)
                logger.info(f"🎤 Using IndicConformer for Indic languages (detected_language='{detected_language}')")
                transcription = self.transcribe_with_indic_conformer(waveform, sr, detected_language, slice_id)
            
            response = TranscriptionResponse(
                success=True,
                request_id=request_id,
                transcription=transcription,
                detected_language=detected_language,
                language_code=detected_language,
                processing_time=asyncio.get_event_loop().time() - start_time
            )
            
            # Store result
            self.results[request_id] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing transcription {request_id}: {str(e)}")
            result = TranscriptionResponse(
                success=False,
                request_id=request_id,
                error=f"Transcription failed: {str(e)}",
                processing_time=asyncio.get_event_loop().time() - start_time
            )
            self.results[request_id] = result
            return result
            
        finally:
            # Always release the GPU slice
            if slice_id:
                gpu_pool.release_slice(slice_id, request_id)
            
            # Clean up request tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]

# Global transcription service
transcription_service = TranscriptionService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global whisper_models, whisper_processors, indic_models, gpu_pool
    
    try:
        # Initialize GPU pool with MIG devices from GPU 1
        # These are the three MIG devices you mentioned
        mig_devices = [
            "MIG-587837bd-78f7-5d96-ad3b-568d0b1febb9",  # MIG 2g.35gb Device 0
            "MIG-a84e0f2e-8d0e-5887-b547-186f9a42c479",  # MIG 2g.35gb Device 1  
            "MIG-3c258235-738f-50c8-8173-6977a3c22ca7"   # MIG 2g.35gb Device 2
        ]
        
        logger.info("Initializing GPU pool...")
        gpu_pool = initialize_gpu_pool(mig_devices)
        
        # Load both Whisper and IndicConformer models for each GPU slice
        logger.info("Loading Whisper and IndicConformer models for each GPU slice...")
        available_devices = torch.cuda.device_count()
        logger.info(f"Available CUDA devices: {available_devices}")
        
        for i, device_id in enumerate(mig_devices):
            slice_id = f"mig_{i}"
            # All MIG devices are on GPU 1, so use cuda:1 for all
            device = "cuda:1"
            
            logger.info(f"Loading models for {slice_id} on {device}...")
            torch.cuda.set_device(device)
            
            # Load Whisper processor and model (for English)
            whisper_model_name = "openai/whisper-small"
            whisper_processors[slice_id] = WhisperProcessor.from_pretrained(whisper_model_name)
            whisper_models[slice_id] = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)
            
            # Load IndicConformer model (for Indian languages - MUCH better quality)
            indic_model_name = "ai4bharat/indic-conformer-600m-multilingual"
            indic_models[slice_id] = AutoModel.from_pretrained(
                indic_model_name, 
                trust_remote_code=True
            ).to(device)
            indic_models[slice_id].eval()
            
            logger.info(f"✅ Both Whisper and IndicConformer models loaded for {slice_id}")
        
        logger.info("Transcription API startup completed successfully!")
        logger.info(f"Loaded {len(whisper_models)} Whisper and {len(indic_models)} IndicConformer models for high-quality multilingual transcription")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language_code: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    logger.info(f"🔍 DEBUG: Transcribe endpoint called with language_code='{language_code}'")
    """
    Transcribe uploaded audio file using GPU pooling.
    
    Args:
        audio: Audio file to transcribe
        language_code: Optional language code (if not provided, will auto-detect)
        
    Returns:
        TranscriptionResponse with transcription results
    """
    if not whisper_models or not whisper_processors or not indic_models or not gpu_pool:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process transcription
            result = await transcription_service.process_transcription(
                request_id=request_id,
                audio_path=temp_file_path,
                language_code=language_code
            )
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get current status of the transcription service"""
    if not gpu_pool:
        return {"error": "GPU pool not initialized"}
    
    pool_status = gpu_pool.get_pool_status()
    
    return {
        "service_status": "running",
        "whisper_models_loaded": len(whisper_models),
        "whisper_processors_loaded": len(whisper_processors),
        "indic_models_loaded": len(indic_models),
        "whisper_models": list(whisper_models.keys()),
        "indic_models": list(indic_models.keys()),
        "gpu_pool_status": pool_status,
        "active_requests": len(transcription_service.active_requests),
        "completed_requests": len(transcription_service.results)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "transcription-api"}

@app.get("/results/{request_id}")
async def get_result(request_id: str):
    """Get transcription result by request ID"""
    if request_id in transcription_service.results:
        return transcription_service.results[request_id]
    else:
        raise HTTPException(status_code=404, detail="Result not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
