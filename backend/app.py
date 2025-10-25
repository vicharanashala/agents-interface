#!/usr/bin/env python3
"""
Voice Recommendation System - FastAPI Backend
Based on the notebook implementation for multilingual voice processing and semantic Q&A search.
"""

import os
import time
import logging
from typing import Optional, List
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import login
import uvicorn
import torch
# Import configuration and services
from config import config
# from services.audio_service import AudioService  # Removed - using external transcription API
from services.search_service import SearchService
from services.punctuation_service import PunctuationService

# Custom formatter for IST timezone
class ISTFormatter(logging.Formatter):
    """Custom formatter to show logs in Indian Standard Time (IST)"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo('Asia/Kolkata'))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime('%Y-%m-%d %H:%M:%S')
            s = f"{s},{int(record.msecs):03d}"
        return s

# Configure logging
log_file = "/app/backend/app_performance.log"
formatter = ISTFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
    force=True  # Force reconfiguration even if logging was already configured
)
logger = logging.getLogger()  # Use root logger to ensure it works
logger.info("✅ Logging is working with IST timezone")

# Initialize FastAPI app
def create_app(config_name='development'):
    app = FastAPI(
        title="Voice Recommendation System API",
        description="API for multilingual voice processing and semantic Q&A search",
        version="1.0.0"
    )
    
    # Load configuration
    app_config = config[config_name]
    app.state.config = app_config
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

app = create_app()

# Global service instances
audio_service = None
search_service = None
mms_lid_language_detector = None
punctuation_service = None

# External Transcription API base URL (FastAPI Whisper-only service)
# Set from config after app initialization
TRANSCRIPTION_API_URL = app.state.config.TRANSCRIPTION_API_URL

def call_external_translation_api(text: str, source_language: str, target_language: str = "English") -> dict:
    """Call external translation API and return its JSON response as dict.

    Args:
        text: Text to translate
        source_language: Source language code
        target_language: Target language (default: English)

    Returns:
        Dict parsed from JSON response of the external API
    """
    try:
        # Get translation API URL from config
        translation_url = app.state.config.TRANSLATION_API_BASE_URL
        headers = app.state.config.TRANSLATION_API_HEADERS.copy()
        
        logger.info(f"Calling external translation API: {translation_url}")
        
        # Prepare request data
        data = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language
        }
        
        resp = requests.post(f"{translation_url}/translate/indic-to-en", json=data, headers=headers, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        
        # Convert the response format to match expected format
        return {
            "success": result.get("status") == "success",
            "translated_text": result.get("translated_text", ""),
            "source_language": result.get("source_language", ""),
            "target_language": result.get("target_language", ""),
            "translation_method": "external_api"
        }
    except Exception as e:
        logger.error(f"Error calling translation API: {e}")
        # Return fallback response
        return {
            "success": False,
            "translated_text": text,
            "source_language": source_language,
            "target_language": target_language,
            "translation_method": "external_api_failed",
            "error": str(e)
        }

def call_external_transcription_api(audio_file_path: str, language_code: Optional[str] = None) -> dict:
    """Call external transcription API and return its JSON response as dict.

    Args:
        audio_file_path: Local path to the audio file to upload
        language_code: Optional language code to force transcription language

    Returns:
        Dict parsed from JSON response of the external API
    """
    with open(audio_file_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_file_path), f, "application/octet-stream")}
        data = {}
        if language_code:
            data["language_code"] = language_code
            logger.info(f"🔍 DEBUG: Calling transcription API with language_code='{language_code}'")
        else:
            logger.info(f"🔍 DEBUG: Calling transcription API with no language_code")
        resp = requests.post(f"{TRANSCRIPTION_API_URL}/transcribe", files=files, data=data, timeout=300)
        resp.raise_for_status()
        return resp.json()

def detect_language_mms_lid(audio_file_path: str) -> dict:
    """Use MMS-LID language detection to detect language before transcription.
    
    Args:
        audio_file_path: Local path to the audio file
        
    Returns:
        Dict with detected language information
    """
    global mms_lid_language_detector
    
    try:
        # Use pre-loaded global hybrid detector instance
        if mms_lid_language_detector is None:
            logger.warning("MMS-LID language detector not initialized - falling back to auto-detect")
            return {
                "success": False,
                "detected_language": "Auto-detect",
                "language_code": None,
                "confidence": 0.0,
                "detection_method": "not_initialized",
                "error": "MMS-LID language detector not initialized"
            }
        
        # Use MMS-LID detection with pre-loaded models
        result = mms_lid_language_detector.detect_language(audio_file_path)
        logger.info(f"MMS-LID language detection result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in MMS-LID language detection: {e}")
        return {
            "success": False,
            "detected_language": "Auto-detect",
            "language_code": None,
            "confidence": 0.0,
            "detection_method": "error_fallback",
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    """Initialize services on FastAPI startup."""
    logger.info("FastAPI startup: Initializing services...")
    if not initialize_services():
        logger.error("Failed to initialize services during startup")
    else:
        logger.info("Services initialized successfully during startup")

def initialize_services():
    """Initialize services for hybrid language detection only."""
    global audio_service, search_service, mms_lid_language_detector, punctuation_service
    
    try:
        app_config = app.state.config
        
        # Skip HF login for now - we're using external transcription API
        logger.info("Skipping HF authentication - using external transcription API")
        
        # Initialize MMS-LID language detection models at startup
        logger.info("Initializing MMS-LID language detection models...")
        try:
            from services.mms_lid_language_detection import MmsLidLanguageDetection
            mms_lid_language_detector = MmsLidLanguageDetection(app_config)
            if not mms_lid_language_detector.initialize_models():
                logger.warning("Failed to initialize MMS-LID language detection models - continuing without it")
                mms_lid_language_detector = None
            else:
                logger.info("✅ MMS-LID language detection models loaded successfully")
        except Exception as e:
            logger.warning(f"MMS-LID language detection initialization failed: {e} - continuing without it")
            mms_lid_language_detector = None
        
        # Skip audio service initialization - using external transcription API only
        logger.info("Skipping audio service - using external transcription API only")
        audio_service = None
        
        # Initialize punctuation service
        logger.info("Initializing punctuation service...")
        try:
            punctuation_service = PunctuationService(app_config)
            if not punctuation_service.initialize_model():
                logger.warning("Failed to initialize punctuation service - continuing without it")
                punctuation_service = None
            else:
                logger.info("✅ Punctuation service loaded successfully")
        except Exception as e:
            logger.warning(f"Punctuation service initialization failed: {e} - continuing without it")
            punctuation_service = None
        
        # Initialize search service  
        logger.info("Initializing search service...")
        try:
            search_service = SearchService(app_config)
            if not search_service.initialize_model():
                logger.warning("Failed to initialize search service model - continuing without it")
                search_service = None
            else:
                # Load Q&A dataset
                if not search_service.load_qa_dataset():
                    logger.warning("Failed to load Q&A dataset - continuing without it")
                    search_service = None
        except Exception as e:
            logger.warning(f"Search service initialization failed: {e} - continuing without it")
            search_service = None
        
        logger.info("All services initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        return False

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# WebSocket endpoints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info(f"Client connected: {websocket.client}")
    
    try:
        await websocket.send_json({"message": "Connected to live audio processing server"})
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_json()
            
            if data.get("action") == "start_live_processing":
                logger.info(f"Client {websocket.client} started live processing")
                await websocket.send_json({
                    "message": "Live processing started", 
                    "timestamp": time.time()
                })
            elif data.get("action") == "stop_live_processing":
                logger.info(f"Client {websocket.client} stopped live processing")
                await websocket.send_json({
                    "message": "Live processing stopped", 
                    "timestamp": time.time()
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
        manager.disconnect(websocket)

async def broadcast_question_result(result_data):
    """Broadcast question processing result to all connected clients."""
    await manager.broadcast({"type": "question_result", "data": result_data})

def allowed_file(filename: str) -> bool:
    """Check if uploaded file has allowed extension."""
    app_config = app.state.config
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app_config.ALLOWED_EXTENSIONS

# API Routes

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        dataset_info = search_service.get_dataset_info() if search_service else None
        
        return {
            "status": "healthy",
            "audio_service_loaded": audio_service is not None,
            "search_service_loaded": search_service is not None,
            "dataset_info": dataset_info
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/dataset-info")
async def get_dataset_info():
    """Get information about the loaded Q&A dataset."""
    if not search_service:
        raise HTTPException(status_code=500, detail="Search service not initialized")
        
    info = search_service.get_dataset_info()
    return {
        "success": True,
        "dataset_info": info
    }

@app.post("/debug-audio")
async def debug_audio(audio: UploadFile = File(...)):
    """
    Debug endpoint to analyze audio file issues.
    
    Expected: multipart/form-data with 'audio' file field
    Returns: JSON with detailed audio file information
    """
    try:
        # audio_service is optional - we use external transcription API
        
        # Create uploads directory if it doesn't exist
        app_config = app.state.config
        os.makedirs(app_config.UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded file
        filename = audio.filename
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        try:
            # Simple file validation - just check if file exists and has size
            import os
            file_size = os.path.getsize(filepath)
            is_valid = file_size > 0
            
            return {
                "success": True,
                "filename": filename,
                "file_info": {
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                },
                "validation": {
                    "is_valid": is_valid
                },
                "preprocessing_test": {
                    "success": True,
                    "note": "Using external transcription API - no local preprocessing needed"
                }
            }
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        logger.error(f"Error in debug-audio endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/transcribe")
async def transcribe_audio(
    request: Request,
    audio: UploadFile = File(...),
    translate: bool = Form(default=True)
):
    """
    Transcribe uploaded audio file.
    
    Expected: multipart/form-data with:
    - 'audio' file field (required)
    - 'translate' boolean field (optional, defaults to True)
    
    Returns: JSON with transcription results
    """
    try:
        # Create uploads directory if it doesn't exist
        app_config = app.state.config
        os.makedirs(app_config.UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded file
        filename = audio.filename
        if not allowed_file(filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {', '.join(app_config.ALLOWED_EXTENSIONS)}"
            )
        
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        try:
            # Step 1: Detect language using hybrid language detection
            lang_result = detect_language_mms_lid(filepath)
            
            if not lang_result['success']:
                raise HTTPException(status_code=500, detail=f"Language detection failed: {lang_result.get('error', 'Unknown error')}")
            
            detected_language_code = lang_result['language_code']
            detected_language_name = lang_result['detected_language']
            confidence = lang_result.get('confidence', 0.0)
            
            # Step 2: Transcribe via external transcription API with detected language
            logger.info(f"🔍 DEBUG: Sending to transcription API - detected_language_code='{detected_language_code}'")
            result = call_external_transcription_api(filepath, language_code=detected_language_code)
            
            if result['success']:
                # Step 3: Add punctuation to the transcription (skip for English)
                raw_transcription = result.get('transcription', '')
                if punctuation_service and detected_language_code != 'en':
                    logger.info(f"Applying punctuation for {detected_language_name} ({detected_language_code})")
                    punctuation_result = punctuation_service.punctuate_text(raw_transcription)
                else:
                    if detected_language_code == 'en':
                        logger.info("Skipping punctuation for English text")
                    punctuation_result = {'punctuated_text': raw_transcription, 'success': True}  # Skip punctuation for English or fallback
                
                # Prepare base response
                response = {
                    "success": True,
                    "metadata": {
                        "audio_file": filename,
                        "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                        "processing_timestamp": time.time()
                    },
                    "language_detection": {
                        "detected_language": detected_language_name,
                        "language_code": detected_language_code,
                        "lid_code": lang_result.get('lid_code', ''),
                        "confidence": confidence,
                        "detection_method": lang_result.get('detection_method', 'hybrid'),
                        "model_used": lang_result.get('model_used', 'hybrid')
                    },
                    "transcription": {
                        "original_text": raw_transcription,
                        "language": result.get('detected_language', ''),
                        "character_count": len(raw_transcription),
                        "word_count": len(raw_transcription.split())
                    },
                    "punctuation": {
                        "success": punctuation_result['success'],
                        "punctuated_text": punctuation_result.get('punctuated_text', raw_transcription),
                        "character_count": punctuation_result.get('character_count', len(raw_transcription)),
                        "word_count": punctuation_result.get('word_count', len(raw_transcription.split())),
                        "punctuation_added": punctuation_result.get('punctuation_added', 0),
                        "error": punctuation_result.get('error', '') if not punctuation_result['success'] else None
                    }
                }
                
                # Add translation only if requested
                if translate:
                    text_to_translate = response["punctuation"]["punctuated_text"]
                    # Use external translation API
                    translation_result = call_external_translation_api(
                        text_to_translate, 
                        response["language_detection"].get("language_code", ""),
                        "English"
                    )
                    response["translation"] = {
                        "success": translation_result['success'],
                        "translated_text": translation_result.get('translated_text', ''),
                        "source_language": translation_result.get('source_language', ''),
                        "target_language": translation_result.get('target_language', ''),
                        "translation_method": translation_result.get('translation_method', ''),
                        "note": translation_result.get('note', ''),
                        "source_text_used": text_to_translate,
                        "error": translation_result.get('error', '') if not translation_result['success'] else None
                    }
                else:
                    response["translation"] = {
                        "success": False,
                        "translated_text": "",
                        "source_language": "",
                        "target_language": "",
                        "translation_method": "skipped",
                        "note": "Translation was skipped as requested",
                        "source_text_used": "",
                        "error": None
                    }
                
                return response
            else:
                raise HTTPException(status_code=500, detail=result['error'])
                
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

@app.post("/punctuate")
async def punctuate_text(request: TextRequest):
    """
    Add punctuation to text.
    
    Expected: JSON with 'text' field
    Returns: JSON with punctuation results
    """
    try:
        if not audio_service:
            raise HTTPException(status_code=500, detail="Audio service not initialized")
        
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text provided")
            
        # Simple language detection for punctuation endpoint
        def is_english_text(text: str) -> bool:
            """Simple heuristic to detect if text is primarily English."""
            if not text:
                return True
            # Check if text contains mostly Latin characters
            latin_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
            total_chars = sum(1 for c in text if c.isalpha())
            if total_chars == 0:
                return True
            return (latin_chars / total_chars) > 0.9
        
        # Use punctuation service (skip for English)
        if punctuation_service and not is_english_text(text):
            logger.info("Applying punctuation for non-English text")
            result = punctuation_service.punctuate_text(text)
        else:
            if is_english_text(text):
                logger.info("Skipping punctuation for English text")
            result = {'punctuated_text': text, 'success': True}  # Skip punctuation for English or fallback
        
        return {
            "success": result['success'],
            "original_text": result.get('original_text', text),
            "punctuated_text": result.get('punctuated_text', text),
            "character_count": result.get('character_count', len(text)),
            "word_count": result.get('word_count', len(text.split())),
            "punctuation_added": result.get('punctuation_added', 0),
            "error": result.get('error', '') if not result['success'] else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in punctuate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

class SearchRequest(BaseModel):
    question: str
    similarity_threshold: Optional[float] = None
    top_k: Optional[int] = None

@app.post("/search")
async def search_qa(request: SearchRequest):
    """
    Search Q&A dataset using semantic similarity.
    
    Expected: JSON with 'question' field
    Returns: JSON with search results
    """
    try:
        if not search_service:
            raise HTTPException(status_code=500, detail="Search service not initialized")
        
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Empty question provided")
            
        # Perform semantic search using search service
        result = search_service.semantic_search(
            question, 
            request.similarity_threshold, 
            request.top_k
        )
        
        return {
            "success": True,
            "query": question,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

class KeywordSearchRequest(BaseModel):
    keywords: List[str]
    top_k: Optional[int] = 5

@app.post("/search-keywords")
async def search_by_keywords(request: KeywordSearchRequest):
    """
    Search Q&A dataset using keywords.
    
    Expected: JSON with 'keywords' field (array of strings)
    Returns: JSON with search results
    """
    try:
        if not search_service:
            raise HTTPException(status_code=500, detail="Search service not initialized")
        
        keywords = request.keywords
        if not keywords:
            raise HTTPException(status_code=400, detail="Keywords must be a non-empty array")
            
        # Perform keyword search
        result = search_service.search_by_keywords(keywords, request.top_k)
        
        return {
            "success": True,
            "keywords": keywords,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in keyword search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-audio")
async def process_audio_complete(audio: UploadFile = File(...)):
    """
    Complete workflow: transcribe audio and perform semantic search.
    
    Expected: multipart/form-data with 'audio' file field
    Returns: JSON with transcription and search results
    """
    # Start overall timing
    start_time = time.time()
    timing_info = {}
    
    try:
        # Print device info
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.get_device_properties(0))
        print(torch.cuda.get_device_capability(0))
        print(torch.cuda.get_device_properties(0))
        print(torch.cuda.get_device_properties(0))
       
        if not search_service:
            raise HTTPException(status_code=500, detail="Search service not initialized")
        
        # Create uploads directory if it doesn't exist
        app_config = app.state.config
        os.makedirs(app_config.UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded file
        file_save_start = time.time()
        filename = audio.filename
        if not allowed_file(filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {', '.join(app_config.ALLOWED_EXTENSIONS)}"
            )
        
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        file_save_time = time.time() - file_save_start
        timing_info['file_save_time'] = round(file_save_time, 3)
        logger.info(f"⏱️  File save time: {file_save_time:.3f}s")
        
        try:
            # Step 1: Language Detection
            language_detection_start = time.time()
            lang_result = detect_language_mms_lid(filepath)
            language_detection_time = time.time() - language_detection_start
            timing_info['language_detection_time'] = round(language_detection_time, 3)
            logger.info(f"⏱️  Language detection time: {language_detection_time:.3f}s")
            
            if not lang_result['success']:
                raise HTTPException(status_code=500, detail=f"Language detection failed: {lang_result.get('error', 'Unknown error')}")
            
            detected_language_code = lang_result['language_code']
            detected_language_name = lang_result['detected_language']
            confidence = lang_result.get('confidence', 0.0)
            
            # Step 2: Transcribe via external transcription API with detected language
            transcription_start = time.time()
            transcription_result = call_external_transcription_api(filepath, language_code=detected_language_code)
            transcription_time = time.time() - transcription_start
            timing_info['transcription_time'] = round(transcription_time, 3)
            logger.info(f"⏱️  Transcription time: {transcription_time:.3f}s")
            
            if not transcription_result.get('success', False):
                raise HTTPException(
                    status_code=500, 
                    detail=f"Transcription failed: {transcription_result.get('error', 'unknown')}"
                )
            
            # Step 2.5: Add punctuation to the transcription using the audio service (skip for English)
            punctuation_start = time.time()
            original_text = transcription_result.get('transcription', '')
            if punctuation_service and detected_language_code != 'en':
                logger.info(f"Applying punctuation for {detected_language_name} ({detected_language_code})")
                punctuation_result = punctuation_service.punctuate_text(original_text)
            else:
                if detected_language_code == 'en':
                    logger.info("Skipping punctuation for English text")
                punctuation_result = {'punctuated_text': original_text, 'success': True}  # Skip punctuation for English or fallback
            punctuation_time = time.time() - punctuation_start
            timing_info['punctuation_time'] = round(punctuation_time, 3)
            logger.info(f"⏱️  Punctuation time: {punctuation_time:.3f}s")
            
            # Step 3: Translate the punctuated text to English
            translation_start = time.time()
            text_to_translate = punctuation_result.get('punctuated_text', original_text)
            # Use external translation API
            translation_result = call_external_translation_api(
                text_to_translate, 
                detected_language_code,
                "English"
            )
            translation_time = time.time() - translation_start
            timing_info['translation_time'] = round(translation_time, 3)
            logger.info(f"⏱️  Translation time: {translation_time:.3f}s")
            
            # Step 3: Perform semantic search on translated text using search service
            search_start = time.time()
            search_text = translation_result['translated_text'] if translation_result['success'] else transcription_result['transcription']
            search_result = search_service.semantic_search(search_text)
            search_time = time.time() - search_start
            timing_info['search_time'] = round(search_time, 3)
            logger.info(f"⏱️  Search time: {search_time:.3f}s")
            
            # Calculate total processing time
            total_time = time.time() - start_time
            timing_info['total_time'] = round(total_time, 3)
            logger.info(f"⏱️  Total processing time: {total_time:.3f}s")
            
            return {
                "success": True,
                "metadata": {
                    "audio_file": filename,
                    "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                    "processing_timestamp": time.time()
                },
                "timing": timing_info,
                "language_detection": {
                    "detected_language": detected_language_name,
                    "language_code": detected_language_code,
                    "lid_code": lang_result.get('lid_code', ''),
                    "confidence": confidence,
                    "detection_method": lang_result.get('detection_method', 'hybrid'),
                    "model_used": lang_result.get('model_used', 'hybrid')
                },
                "transcription": {
                    "original_text": original_text,
                    "language": detected_language_name,
                    "character_count": len(original_text or ''),
                    "word_count": len((original_text or '').split())
                },
                "punctuation": {
                    "success": punctuation_result['success'],
                    "punctuated_text": punctuation_result.get('punctuated_text', original_text),
                    "character_count": punctuation_result.get('character_count', len(original_text)),
                    "word_count": punctuation_result.get('word_count', len(original_text.split())),
                    "punctuation_added": punctuation_result.get('punctuation_added', 0),
                    "error": punctuation_result.get('error', '') if not punctuation_result['success'] else None
                },
                "translation": {
                    "success": translation_result['success'],
                    "translated_text": translation_result.get('translated_text', ''),
                    "source_language": translation_result.get('source_language', ''),
                    "target_language": translation_result.get('target_language', ''),
                    "translation_method": translation_result.get('translation_method', ''),
                    "note": translation_result.get('note', ''),
                    "source_text_used": text_to_translate,
                    "error": translation_result.get('error', '') if not translation_result['success'] else None
                },
                "semantic_search": {
                    "query_used": search_text,
                    "search_result": search_result
                }
            }
            
        finally:
            # Clean up uploaded file
            cleanup_start = time.time()
            if os.path.exists(filepath):
                os.remove(filepath)
            cleanup_time = time.time() - cleanup_start
            timing_info['cleanup_time'] = round(cleanup_time, 3)
            logger.info(f"⏱️  Cleanup time: {cleanup_time:.3f}s")
            
            # Log final timing summary
            total_time = time.time() - start_time
            logger.info(f"⏱️  FINAL SUMMARY:")
            logger.info(f"   📁 File save: {timing_info.get('file_save_time', 0):.3f}s")
            logger.info(f"   🔍 Language detection: {timing_info.get('language_detection_time', 0):.3f}s")
            logger.info(f"   🎤 Transcription: {timing_info.get('transcription_time', 0):.3f}s")
            logger.info(f"   📝 Punctuation: {timing_info.get('punctuation_time', 0):.3f}s")
            logger.info(f"   🌐 Translation: {timing_info.get('translation_time', 0):.3f}s")
            logger.info(f"   🔍 Search: {timing_info.get('search_time', 0):.3f}s")
            logger.info(f"   🧹 Cleanup: {timing_info.get('cleanup_time', 0):.3f}s")
            logger.info(f"   ⏱️  TOTAL: {total_time:.3f}s")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process-audio endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-live-audio")
async def process_live_audio(
    audio: UploadFile = File(...),
    timestamp: str = Form(default_factory=lambda: str(time.time()))
):
    """
    Process live audio chunks with real-time transcription and search.
    
    Expected: multipart/form-data with 'audio' file field and 'timestamp' field
    Returns: JSON with transcription and search results
    """
    try:
        if not audio_service or not search_service:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        # Create uploads directory if it doesn't exist
        app_config = app.state.config
        os.makedirs(app_config.UPLOAD_FOLDER, exist_ok=True)
        
        # Save uploaded file
        filename = audio.filename
        if not allowed_file(filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {', '.join(app_config.ALLOWED_EXTENSIONS)}"
            )
        
        filepath = os.path.join(app_config.UPLOAD_FOLDER, filename)
        
        with open(filepath, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        try:
            # Step 1: Detect language and transcribe using external API
            lang_result = detect_language_mms_lid(filepath)
            language_code = lang_result.get('language_code', 'auto')
            transcription_result = call_external_transcription_api(filepath, language_code)
            
            if not transcription_result['success']:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Transcription failed: {transcription_result['error']}"
                )
            
            # Step 2: Add punctuation to the transcription (skip for English)
            original_text = transcription_result.get('transcription', '')
            detected_language_code = transcription_result.get('language_code', 'auto')
            if punctuation_service and detected_language_code != 'en':
                logger.info(f"Applying punctuation for {transcription_result.get('detected_language', 'unknown')} ({detected_language_code})")
                punctuation_result = punctuation_service.punctuate_text(original_text)
            else:
                if detected_language_code == 'en':
                    logger.info("Skipping punctuation for English text")
                punctuation_result = {'punctuated_text': original_text, 'success': True}  # Skip punctuation for English or fallback
            
            # Step 3: Translate the punctuated text to English
            text_to_translate = punctuation_result.get('punctuated_text', original_text)
            # Use external translation API
            translation_result = call_external_translation_api(
                text_to_translate, 
                transcription_result['language_code'],
                "English"
            )
            
            # Step 3: Perform semantic search on translated text using search service
            search_text = translation_result['translated_text'] if translation_result['success'] else transcription_result['transcription']
            search_result = search_service.semantic_search(search_text)
            
            result_data = {
                "success": True,
                "timestamp": timestamp,
                "metadata": {
                    "audio_file": filename,
                    "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                    "processing_timestamp": time.time()
                },
                "language_detection": {
                    "detected_language": transcription_result['detected_language'],
                    "language_code": transcription_result['language_code'],
                    "lid_code": transcription_result.get('lid_code', ''),
                    "confidence": "High"
                },
                "transcription": {
                    "original_text": transcription_result['transcription'],
                    "language": transcription_result['detected_language'],
                    "character_count": len(transcription_result['transcription']),
                    "word_count": len(transcription_result['transcription'].split())
                },
                "punctuation": {
                    "success": punctuation_result['success'],
                    "punctuated_text": punctuation_result.get('punctuated_text', original_text),
                    "character_count": punctuation_result.get('character_count', len(original_text)),
                    "word_count": punctuation_result.get('word_count', len(original_text.split())),
                    "punctuation_added": punctuation_result.get('punctuation_added', 0),
                    "error": punctuation_result.get('error', '') if not punctuation_result['success'] else None
                },
                "translation": {
                    "success": translation_result['success'],
                    "translated_text": translation_result.get('translated_text', ''),
                    "source_language": translation_result.get('source_language', ''),
                    "target_language": translation_result.get('target_language', ''),
                    "translation_method": translation_result.get('translation_method', ''),
                    "note": translation_result.get('note', ''),
                    "source_text_used": text_to_translate,
                    "error": translation_result.get('error', '') if not translation_result['success'] else None
                },
                "semantic_search": {
                    "query_used": search_text,
                    "search_result": search_result
                }
            }
            
            # Broadcast the result via WebSocket
            await broadcast_question_result(result_data)
            
            return result_data
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process-live-audio endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.exception_handler(413)
async def too_large_handler(request, exc):
    """Handle file too large error."""
    return JSONResponse(
        status_code=413,
        content={"success": False, "error": "File too large. Maximum size is 50MB."}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors."""
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error occurred."}
    )

if __name__ == '__main__':
    logger.info("Initializing Voice Recommendation System API...")
    
    # Initialize services
    if not initialize_services():
        logger.error("Failed to initialize services. Exiting.")
        exit(1)
    
    logger.info("All systems initialized successfully!")
    
    # Run the FastAPI app with uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5020,
        reload=True,
        log_level="info",
        workers=1  
    )

    #uvicorn app:app --host 0.0.0.0 --port 5000 --workers 2
