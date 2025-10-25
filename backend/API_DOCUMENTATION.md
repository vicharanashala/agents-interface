# Voice Recommendation System - Backend API Documentation

## Overview

The Voice Recommendation System is a FastAPI-based backend service that provides multilingual voice processing and semantic Q&A search capabilities. The system supports audio transcription, language detection, punctuation, translation, and semantic search across multiple Indian languages and English.

## Architecture

### Core Components

1. **FastAPI Application** (`app.py`) - Main application with REST API endpoints
2. **Services Layer** - Modular services for different functionalities
3. **Configuration** (`config.py`) - Centralized configuration management
4. **External APIs** - Integration with transcription and translation services

### Service Architecture

```
app.py (FastAPI Application)
├── services/
│   ├── mms_lid_language_detection.py  # Language detection
│   ├── punctuation_service.py    # Text punctuation
│   └── search_service.py        # Semantic search
├── config.py                    # Configuration management
└── External APIs
    ├── Transcription API        # External Whisper-based service
    └── Translation API          # External translation service
```

## API Endpoints

### Health and Information Endpoints

#### `GET /health`
**Description**: Health check endpoint
**Response**:
```json
{
  "status": "healthy",
  "audio_service_loaded": false,
  "search_service_loaded": true,
  "dataset_info": {
    "loaded": true,
    "total_pairs": 1000,
    "faiss_index_size": 1000
  }
}
```

#### `GET /dataset-info`
**Description**: Get information about the loaded Q&A dataset
**Response**:
```json
{
  "success": true,
  "dataset_info": {
    "loaded": true,
    "total_pairs": 1000,
    "faiss_index_size": 1000,
    "avg_question_length": 15.5,
    "avg_answer_length": 45.2,
    "sample_questions": ["What is paddy?", "How to grow rice?"]
  }
}
```

### Audio Processing Endpoints

#### `POST /transcribe`
**Description**: Transcribe uploaded audio file with language detection and punctuation
**Request**: `multipart/form-data`
- `audio`: Audio file (required)
- `translate`: Boolean (optional, defaults to true)

**Response**:
```json
{
  "success": true,
  "metadata": {
    "audio_file": "audio.wav",
    "file_size": 1024000,
    "processing_timestamp": 1640995200.0
  },
  "language_detection": {
    "detected_language": "Hindi",
    "language_code": "hi",
    "confidence": 0.95,
    "detection_method": "hybrid",
    "model_used": "mms_lid"
  },
  "transcription": {
    "original_text": "मेरा नाम राम है",
    "language": "Hindi",
    "character_count": 15,
    "word_count": 3
  },
  "punctuation": {
    "success": true,
    "punctuated_text": "मेरा नाम राम है।",
    "character_count": 16,
    "word_count": 3,
    "punctuation_added": 1
  },
  "translation": {
    "success": true,
    "translated_text": "My name is Ram.",
    "source_language": "Hindi",
    "target_language": "English",
    "translation_method": "external_api"
  }
}
```

#### `POST /process-audio`
**Description**: Complete workflow - transcribe audio and perform semantic search
**Request**: `multipart/form-data`
- `audio`: Audio file (required)

**Response**: Includes all transcription data plus semantic search results:
```json
{
  "success": true,
  "timing": {
    "file_save_time": 0.001,
    "language_detection_time": 0.5,
    "transcription_time": 2.1,
    "punctuation_time": 0.3,
    "translation_time": 1.2,
    "search_time": 0.4,
    "total_time": 4.5
  },
  "language_detection": { /* ... */ },
  "transcription": { /* ... */ },
  "punctuation": { /* ... */ },
  "translation": { /* ... */ },
  "semantic_search": {
    "query_used": "My name is Ram.",
    "search_result": {
      "status": "success",
      "results": [
        {
          "rank": 1,
          "answer": "Ram is a common Indian name...",
          "matched_question": "What is the meaning of Ram?",
          "distance": 0.15,
          "similarity_score": 0.87,
          "confidence": "High"
        }
      ]
    }
  }
}
```

#### `POST /process-live-audio`
**Description**: Process live audio chunks with real-time WebSocket broadcasting
**Request**: `multipart/form-data`
- `audio`: Audio file (required)
- `timestamp`: Timestamp string (optional)

**Response**: Same as `/process-audio` but also broadcasts results via WebSocket

#### `POST /debug-audio`
**Description**: Debug endpoint to analyze audio file issues
**Request**: `multipart/form-data`
- `audio`: Audio file (required)

**Response**:
```json
{
  "success": true,
  "filename": "audio.wav",
  "file_info": {
    "size_bytes": 1024000,
    "size_mb": 1.0
  },
  "validation": {
    "is_valid": true
  },
  "preprocessing_test": {
    "success": true,
    "note": "Using external transcription API - no local preprocessing needed"
  }
}
```

### Text Processing Endpoints

#### `POST /punctuate`
**Description**: Add punctuation to text
**Request**: JSON
```json
{
  "text": "मेरा नाम राम है"
}
```

**Response**:
```json
{
  "success": true,
  "original_text": "मेरा नाम राम है",
  "punctuated_text": "मेरा नाम राम है।",
  "character_count": 16,
  "word_count": 3,
  "punctuation_added": 1
}
```

### Search Endpoints

#### `POST /search`
**Description**: Semantic search using Q&A dataset
**Request**: JSON
```json
{
  "question": "What is paddy cultivation?",
  "similarity_threshold": 0.3,
  "top_k": 5
}
```

**Response**:
```json
{
  "success": true,
  "query": "What is paddy cultivation?",
  "result": {
    "status": "success",
    "results": [
      {
        "rank": 1,
        "answer": "Paddy cultivation is the process of growing rice...",
        "matched_question": "How to grow paddy?",
        "distance": 0.12,
        "similarity_score": 0.89,
        "confidence": "High"
      }
    ],
    "total_results": 5
  }
}
```

#### `POST /search-keywords`
**Description**: Search Q&A dataset using keywords
**Request**: JSON
```json
{
  "keywords": ["paddy", "rice", "cultivation"],
  "top_k": 5
}
```

**Response**:
```json
{
  "success": true,
  "keywords": ["paddy", "rice", "cultivation"],
  "result": {
    "status": "success",
    "results": [
      {
        "question": "What is paddy cultivation?",
        "answer": "Paddy cultivation involves...",
        "keyword_matches": 3,
        "index": 42
      }
    ],
    "total_matches": 15
  }
}
```


## Services Documentation

### 1. MmsLidLanguageDetection Service

**File**: `services/mms_lid_language_detection.py`

**Purpose**: Hybrid language detection using Whisper and Facebook MMS-LID models

**Key Methods**:
- `initialize_models()`: Load Whisper and MMS-LID models
- `detect_language_hybrid(audio_path)`: Main detection method
- `detect_language_with_whisper(audio_path)`: Whisper-based detection
- `detect_language_with_mms_lid(audio_path)`: MMS-LID detection

**Supported Languages**:
- English (en)
- Hindi (hi), Bengali (bn), Telugu (te), Tamil (ta)
- Gujarati (gu), Kannada (kn), Malayalam (ml), Marathi (mr)
- Punjabi (pa), Odia (or), Assamese (as), Urdu (ur)
- Nepali (ne), Sanskrit (sa), Sindhi (sd)
- And more Indian languages

**Detection Strategy**:
1. Use Whisper first for language detection
2. If English detected, return English
3. If Indian language with high confidence (>80%), return that language
4. Otherwise, fallback to MMS-LID model

### 2. PunctuationService

**File**: `services/punctuation_service.py`

**Purpose**: Add punctuation to transcribed text using Cadence model

**Key Methods**:
- `initialize_model()`: Load Cadence punctuation model
- `punctuate_text(text)`: Add punctuation to input text
- `is_initialized()`: Check if service is ready

**Features**:
- Uses AI4Bharat's Cadence model
- GPU acceleration support
- Handles multiple Indian languages
- Skips punctuation for English text

### 3. SearchService

**File**: `services/search_service.py`

**Purpose**: Semantic search using FAISS index and sentence transformers

**Key Methods**:
- `initialize_model()`: Load sentence transformer model
- `load_qa_dataset()`: Load FAISS index and metadata
- `semantic_search(question, threshold, top_k)`: Perform semantic search
- `search_by_keywords(keywords, top_k)`: Keyword-based search
- `get_dataset_info()`: Get dataset statistics

**Features**:
- FAISS-based vector search
- Sentence transformer embeddings
- Similarity scoring and ranking
- Keyword matching capabilities


## Configuration

### Environment Variables

```bash
# Hugging Face Token
HF_TOKEN=hf_your_token_here

# Translation API
TRANSLATION_API_BASE_URL=https://your-translation-api.com

# Transcription API
TRANSCRIPTION_API_URL=http://localhost:8000

# Ngrok (for tunneling)
NGROK_DOMAIN=your-domain.ngrok-free.dev
```

### Configuration Classes

**DevelopmentConfig**: Development settings with debug enabled
**ProductionConfig**: Production settings with security considerations
**TestingConfig**: Testing settings with reduced file size limits

### Model Configuration

```python
# Sentence Transformer
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

# Language Detection
LID_MODEL_ID = "facebook/mms-lid-126"

# Punctuation
PUNCTUATION_MODEL_ID = 'ai4bharat/Cadence'

# Search Settings
DEFAULT_TOP_K = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.3
```

## Data Flow

### Complete Audio Processing Pipeline

1. **Audio Upload** → File validation and temporary storage
2. **Language Detection** → MMS-LID hybrid detection
3. **Transcription** → External Whisper-based API
4. **Punctuation** → Cadence model (skip for English)
5. **Translation** → External translation API (if not English)
6. **Semantic Search** → FAISS-based vector search
7. **Response** → Structured JSON with all results

### Real-time Processing

1. **Audio Chunks** → Continuous audio processing via `/process-live-audio`
2. **Live Updates** → Real-time transcription and search results

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid file, empty text)
- `413`: File Too Large (exceeds 50MB limit)
- `500`: Internal Server Error

### Error Response Format

```json
{
  "success": false,
  "error": "Detailed error message",
  "error_type": "validation_error|processing_error|service_error"
}
```

### Service Initialization Errors

- Services gracefully degrade if initialization fails
- System continues to operate with reduced functionality
- Health check endpoint reports service status

## Performance Considerations

### Timing Information

The `/process-audio` endpoint provides detailed timing information:

```json
{
  "timing": {
    "file_save_time": 0.001,
    "language_detection_time": 0.5,
    "transcription_time": 2.1,
    "punctuation_time": 0.3,
    "translation_time": 1.2,
    "search_time": 0.4,
    "total_time": 4.5
  }
}
```

### Optimization Features

- GPU acceleration for ML models
- FAISS index for fast vector search
- External API integration for scalability
- File cleanup after processing

## Deployment

### Running the Application

```bash
# Development
python app.py

# Production with uvicorn
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 2
```

### Docker Support

The system is designed to work in containerized environments with proper volume mounts for:
- Model files
- FAISS indices
- Upload directories
- Log files

### External Dependencies

- **Transcription API**: External Whisper-based service
- **Translation API**: External translation service
- **Model Files**: Downloaded on first run
- **FAISS Index**: Pre-built vector index

## Monitoring and Logging

### Log Files

- `app_performance.log`: Application performance logs
- `backend.log`: General application logs
- `transcription_api.log`: Transcription service logs

### Health Monitoring

- `/health` endpoint for service status
- Service initialization tracking
- Dataset loading verification
- Model availability checks

## Security Considerations

### File Upload Security

- File type validation
- File size limits (50MB)
- Temporary file cleanup
- Allowed extensions whitelist

### API Security

- CORS configuration
- Request validation
- Error message sanitization
- Rate limiting (recommended for production)

## Troubleshooting

### Common Issues

1. **Service Initialization Failures**
   - Check model file availability
   - Verify GPU/CUDA setup
   - Review log files for specific errors

2. **Audio Processing Issues**
   - Validate audio file format
   - Check file size limits
   - Verify external API connectivity

3. **Search Service Issues**
   - Ensure FAISS index is loaded
   - Check metadata file availability
   - Verify sentence transformer model

### Debug Endpoints

- `/debug-audio`: Analyze audio file issues
- `/health`: Check service status
- `/dataset-info`: Verify dataset loading

## API Usage Examples

### cURL Examples

```bash
# Health check
curl -X GET http://localhost:5000/health

# Transcribe audio
curl -X POST -F "audio=@audio.wav" -F "translate=true" \
  http://localhost:5000/transcribe

# Search Q&A
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is paddy cultivation?"}' \
  http://localhost:5000/search

# Punctuate text
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "मेरा नाम राम है"}' \
  http://localhost:5000/punctuate
```

### Python Client Example

```python
import requests

# Transcribe audio
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/transcribe',
        files={'audio': f},
        data={'translate': True}
    )
    result = response.json()
    print(result['transcription']['original_text'])
```


---

This documentation provides a comprehensive overview of the Voice Recommendation System backend API. For specific implementation details, refer to the source code in the respective service files.
