# Transcription API with GPU Pooling

A high-performance FastAPI service for audio transcription using dual-model architecture (Whisper + IndicConformer) with GPU/MIG resource pooling. This service manages three MIG devices on GPU 1 and intelligently selects the optimal model based on language detection.

## Features

- **Dual Model Architecture**: Uses Whisper for English and IndicConformer for Indian languages
- **Intelligent Model Selection**: Automatically chooses the best model based on language detection
- **GPU Pool Management**: Automatically manages three MIG devices on GPU 1
- **Request Routing**: Assigns each transcription request to a free GPU slice
- **Resource Cleanup**: Automatically frees GPU slices after inference
- **Advanced Audio Preprocessing**: Handles multiple audio formats with robust error recovery
- **Language Detection**: Auto-detects language or accepts specified language codes
- **Async Processing**: Handles multiple concurrent requests efficiently
- **High Accuracy**: IndicConformer provides superior quality for Indian languages

## Model Architecture

The service uses a dual-model approach for optimal transcription quality:

### Whisper Model
- **Model**: `openai/whisper-small`
- **Usage**: Primary model for English transcription
- **Processor**: WhisperProcessor for feature extraction
- **Device**: All MIG devices on GPU 1

### IndicConformer Model  
- **Model**: `ai4bharat/indic-conformer-600m-multilingual`
- **Usage**: High-quality transcription for Indian languages (Hindi, Marathi, Bengali, Tamil, etc.)
- **Architecture**: Conformer-based transformer optimized for Indic languages
- **Device**: All MIG devices on GPU 1

### Model Selection Logic
- **English**: Automatically uses Whisper for optimal English transcription
- **Indian Languages**: Uses IndicConformer for superior accuracy and language-specific optimization
- **Fallback**: If IndicConformer fails, automatically falls back to Whisper

## GPU Configuration

The service is configured to use three MIG devices on GPU 1:
- MIG-587837bd-78f7-5d96-ad3b-568d0b1febb9 (MIG 2g.35gb Device 0)
- MIG-a84e0f2e-8d0e-5887-b547-186f9a42c479 (MIG 2g.35gb Device 1)
- MIG-3c258235-738f-50c8-8173-6977a3c22ca7 (MIG 2g.35gb Device 2)

Each GPU slice loads both models simultaneously for instant model switching.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment:
```bash
export CUDA_VISIBLE_DEVICES=1
```

## Usage

### Start the Service

```bash
python run_transcription_api.py
```

The service will start on `http://localhost:8000`

### API Endpoints

#### POST /transcribe
Transcribe an audio file using intelligent model selection.

**Parameters:**
- `audio`: Audio file (multipart/form-data) - supports WAV, MP3, FLAC, OGG, M4A, AAC, MP4, WMA, AMR, AIFF, AU, 3GP, WEBM, MPEG
- `language_code`: Optional language code (e.g., "en", "hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa", "or", "as")

**Model Selection:**
- If `language_code` is "en" or "eng" → Uses Whisper model
- If `language_code` is any Indian language → Uses IndicConformer model  
- If no `language_code` provided → Auto-detects and selects appropriate model

**Response:**
```json
{
  "success": true,
  "request_id": "uuid-4-string",
  "transcription": "transcribed text",
  "detected_language": "hi",
  "language_code": "hi", 
  "confidence": null,
  "processing_time": 2.3,
  "error": null
}
```

**Error Response:**
```json
{
  "success": false,
  "request_id": "uuid-4-string",
  "transcription": null,
  "detected_language": null,
  "language_code": null,
  "confidence": null,
  "processing_time": 1.2,
  "error": "No free GPU slice available"
}
```

#### GET /status
Get comprehensive service and GPU pool status.

**Response:**
```json
{
  "service_status": "running",
  "whisper_models_loaded": 3,
  "whisper_processors_loaded": 3,
  "indic_models_loaded": 3,
  "whisper_models": ["mig_0", "mig_1", "mig_2"],
  "indic_models": ["mig_0", "mig_1", "mig_2"],
  "gpu_pool_status": {
    "total_slices": 3,
    "available_slices": 2,
    "busy_slices": 1,
    "slice_details": {...}
  },
  "active_requests": 1,
  "completed_requests": 15
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "transcription-api"
}
```

#### GET /results/{request_id}
Get transcription result by request ID.

**Response:** Same as POST /transcribe response format

## Testing

Test the API with curl:

```bash
# Test English transcription (uses Whisper)
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_english.wav" \
  -F "language_code=en"

# Test Hindi transcription (uses IndicConformer)
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_hindi.wav" \
  -F "language_code=hi"

# Test auto-detection (no language specified)
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio.wav"

# Check comprehensive status
curl "http://localhost:8000/status"

# Health check
curl "http://localhost:8000/health"

# Get result by request ID
curl "http://localhost:8000/results/{request_id}"
```

## GPU Pool Management

The service automatically:
- Assigns requests to free GPU slices
- Releases slices after processing
- Handles stale assignments (5-minute timeout)
- Provides real-time status monitoring
- Manages error states and recovery

## Audio Processing

### Supported Audio Formats
- **Primary**: WAV, MP3, FLAC, OGG, M4A, AAC, MP4, WMA, AMR, AIFF, AU, 3GP, WEBM, MPEG
- **Automatic Conversion**: All formats are automatically converted to 16kHz mono WAV for optimal processing
- **Robust Loading**: Multiple backend fallbacks (FFmpeg, Sox, SoundFile, Librosa) ensure maximum compatibility

### Audio Preprocessing Pipeline
1. **Format Detection**: Automatically detects audio format and selects appropriate decoder
2. **Channel Conversion**: Converts stereo/multi-channel to mono
3. **Sample Rate Normalization**: Resamples to 16kHz for optimal model performance
4. **Audio Quality Enhancement**: Normalizes volume and fixes common recording issues
5. **Error Recovery**: Multiple fallback strategies for corrupted or unusual audio files

## Language Support

### English (Whisper Model)
- **Primary Model**: OpenAI Whisper Small
- **Languages**: English and 99+ other languages
- **Quality**: Excellent for English, good for international languages

### Indian Languages (IndicConformer Model)
- **Primary Model**: AI4Bharat IndicConformer 600M
- **Optimized For**: Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese
- **Quality**: Superior accuracy for Indian languages compared to Whisper
- **Architecture**: Conformer-based transformer specifically trained on Indian language data

### Model Selection Logic
- **English Detection**: If language_code is "en" or "eng" → Whisper
- **Indian Language Detection**: If language_code is any Indian language code → IndicConformer
- **Auto-Detection**: If no language specified, defaults to IndicConformer (assumes Indian content)
- **Fallback**: If IndicConformer fails, automatically falls back to Whisper

## Technical Implementation Details

### GPU Resource Management
- **MIG Device Pool**: Three MIG slices on GPU 1 for concurrent processing
- **Request Assignment**: Round-robin assignment to available GPU slices
- **Resource Cleanup**: Automatic slice release after processing completion
- **Timeout Handling**: 5-minute timeout for stale assignments
- **Error Recovery**: Graceful handling of GPU errors with automatic retry

### Model Loading Strategy
- **Dual Model Per Slice**: Each GPU slice loads both Whisper and IndicConformer models
- **Memory Optimization**: Models are loaded once per slice and reused for all requests
- **Device Mapping**: All MIG devices mapped to CUDA device 1 for consistent GPU utilization
- **Model Evaluation**: Models set to evaluation mode for optimal inference performance

### Audio Processing Pipeline
- **Multi-Backend Support**: TorchAudio with FFmpeg, Sox, and SoundFile backends
- **Fallback Chain**: Librosa → Audioread for maximum compatibility
- **Quality Enhancement**: Automatic normalization and sample rate conversion
- **Error Handling**: Comprehensive error recovery for corrupted audio files
- **Temporary File Management**: Secure temporary file handling with automatic cleanup

### Performance Optimizations
- **Async Processing**: Non-blocking request handling with asyncio
- **GPU Memory Management**: Efficient GPU memory usage with automatic cleanup
- **Batch Processing**: Optimized model inference with proper tensor handling
- **Request Tracking**: UUID-based request tracking for result retrieval
- **Concurrent Requests**: Support for multiple simultaneous transcription requests

### Error Handling & Monitoring
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Health Checks**: Built-in health monitoring endpoints
- **Status Reporting**: Real-time GPU pool and service status
- **Graceful Degradation**: Automatic fallback mechanisms for model failures
- **Request Timeout**: Configurable timeout handling for long-running requests
