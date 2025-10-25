# Voice Recommendation System - Backend

A FastAPI-based backend service for multilingual voice processing and semantic Q&A search, supporting Indian languages and English.

## 🚀 Features

- **Multilingual Support**: Hindi, Bengali, Telugu, Tamil, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Odia, Assamese, Urdu, Nepali, Sanskrit, Sindhi, and English
- **Hybrid Language Detection**: Combines Whisper and Facebook MMS-LID models
- **Audio Transcription**: External Whisper-based transcription service
- **Text Punctuation**: AI4Bharat's Cadence model for Indian languages
- **Translation**: External translation API for Indic-to-English translation
- **Semantic Search**: FAISS-based vector search with sentence transformers
- **Real-time Processing**: WebSocket support for live audio processing
- **RESTful API**: Comprehensive REST API with detailed documentation

## 📁 Project Structure

```
backend/
├── app.py                          # Main FastAPI application
├── config.py                       # Configuration management
├── requirements.txt                 # Python dependencies
├── requirements_transcription.txt   # Transcription service dependencies
├── services/                       # Service layer
│   ├── __init__.py
│   ├── audio_service.py            # Audio processing (deprecated)
│   ├── mms_lid_language_detection.py  # Language detection
│   ├── punctuation_service.py      # Text punctuation
│   └── search_service.py           # Semantic search
├── API_DOCUMENTATION.md            # Complete API documentation
├── SERVICES_DOCUMENTATION.md      # Services technical documentation
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space for models

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd voice_recommendation_system/backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**
```bash
export HF_TOKEN="your_huggingface_token"
export TRANSCRIPTION_API_URL="http://localhost:8000"
export TRANSLATION_API_BASE_URL="https://your-translation-api.com"
```

5. **Download models** (automatic on first run)
```bash
python -c "from services.mms_lid_language_detection import MmsLidLanguageDetection; from config import DevelopmentConfig; MmsLidLanguageDetection(DevelopmentConfig()).initialize_models()"
```

## 🚀 Quick Start

### Running the Application

```bash
# Development mode
python app.py

# Production mode
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 2
```

### Health Check

```bash
curl http://localhost:5000/health
```

### Basic Usage Examples

#### 1. Transcribe Audio
```bash
curl -X POST -F "audio=@audio.wav" -F "translate=true" \
  http://localhost:5000/transcribe
```

#### 2. Search Q&A
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is paddy cultivation?"}' \
  http://localhost:5000/search
```

#### 3. Add Punctuation
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "मेरा नाम राम है"}' \
  http://localhost:5000/punctuate
```

## 📚 API Documentation

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and service status |
| `/dataset-info` | GET | Q&A dataset information |
| `/transcribe` | POST | Transcribe audio with language detection |
| `/process-audio` | POST | Complete workflow: transcribe + search |
| `/process-live-audio` | POST | Real-time processing with WebSocket |
| `/punctuate` | POST | Add punctuation to text |
| `/search` | POST | Semantic search in Q&A dataset |
| `/search-keywords` | POST | Keyword-based search |
| `/debug-audio` | POST | Debug audio file issues |
| `/ws` | WebSocket | Real-time communication |

### Request/Response Examples

#### Audio Transcription
```json
// Request: POST /transcribe
{
  "audio": "audio_file.wav",
  "translate": true
}

// Response
{
  "success": true,
  "language_detection": {
    "detected_language": "Hindi",
    "language_code": "hi",
    "confidence": 0.95
  },
  "transcription": {
    "original_text": "मेरा नाम राम है",
    "language": "Hindi"
  },
  "punctuation": {
    "punctuated_text": "मेरा नाम राम है।"
  },
  "translation": {
    "translated_text": "My name is Ram."
  }
}
```

#### Semantic Search
```json
// Request: POST /search
{
  "question": "What is paddy cultivation?",
  "top_k": 5
}

// Response
{
  "success": true,
  "result": {
    "status": "success",
    "results": [
      {
        "rank": 1,
        "answer": "Paddy cultivation is the process of growing rice...",
        "matched_question": "How to grow paddy?",
        "similarity_score": 0.89,
        "confidence": "High"
      }
    ]
  }
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Hugging Face
HF_TOKEN=hf_your_token_here

# External APIs
TRANSCRIPTION_API_URL=http://localhost:8000
TRANSLATION_API_BASE_URL=https://your-translation-api.com

# Ngrok (for tunneling)
NGROK_DOMAIN=your-domain.ngrok-free.dev

# Performance
CUDA_VISIBLE_DEVICES=0
MAX_WORKERS=4
```

### Configuration Classes

- **DevelopmentConfig**: Development settings with debug enabled
- **ProductionConfig**: Production settings with security considerations
- **TestingConfig**: Testing settings with reduced limits

### Model Configuration

```python
# Language Detection
LID_MODEL_ID = "facebook/mms-lid-126"
WHISPER_MODEL = "small"

# Punctuation
PUNCTUATION_MODEL_ID = 'ai4bharat/Cadence'

# Search
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_TOP_K = 4
```

## 🏗️ Architecture

### Service Layer

1. **MmsLidLanguageDetection**: Hybrid language detection using Whisper + MMS-LID
2. **PunctuationService**: Text punctuation using Cadence model
3. **SearchService**: Semantic search using FAISS and sentence transformers
4. **AudioService**: Audio processing (deprecated, using external API)

### Data Flow

```
Audio Upload → Language Detection → Transcription → Punctuation → Translation → Semantic Search → Response
```

### External Dependencies

- **Transcription API**: External Whisper-based service
- **Translation API**: External translation service
- **Hugging Face Models**: Language detection and punctuation models
- **FAISS Index**: Pre-built vector search index

## 🧪 Testing

### Health Check
```bash
curl http://localhost:5000/health
```

### Service Status
```bash
curl http://localhost:5000/dataset-info
```

### Audio Debug
```bash
curl -X POST -F "audio=@test.wav" http://localhost:5000/debug-audio
```

### WebSocket Testing
```javascript
const ws = new WebSocket('ws://localhost:5000/ws');
ws.onopen = () => {
    ws.send(JSON.stringify({action: 'start_live_processing'}));
};
```

## 📊 Performance

### Timing Information

The system provides detailed timing information for each processing step:

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
- WebSocket for real-time communication
- Efficient memory management

## 🔒 Security

### File Upload Security
- File type validation (whitelist of allowed extensions)
- File size limits (50MB maximum)
- Temporary file cleanup after processing
- Secure file handling

### API Security
- CORS configuration for cross-origin requests
- Request validation and sanitization
- Error message sanitization
- Rate limiting (recommended for production)

## 🐳 Docker Support

### Dockerfile Example
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "5000:5000"
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - TRANSCRIPTION_API_URL=${TRANSCRIPTION_API_URL}
    volumes:
      - ./models:/app/models
      - ./uploads:/app/uploads
```

## 📝 Logging

### Log Files
- `app_performance.log`: Application performance logs
- `backend.log`: General application logs
- `transcription_api.log`: Transcription service logs

### Log Levels
- **INFO**: General information and status updates
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Critical errors and failures
- **DEBUG**: Detailed debugging information

## 🚨 Troubleshooting

### Common Issues

1. **Service Initialization Failures**
   ```bash
   # Check model files
   ls -la /workspace/voice_recommendation_system/notebook/embedding/paddy/
   
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Audio Processing Issues**
   ```bash
   # Test audio file
   curl -X POST -F "audio=@test.wav" http://localhost:5000/debug-audio
   
   # Check file format
   file test.wav
   ```

3. **Search Service Issues**
   ```bash
   # Check FAISS index
   python -c "import faiss; print(faiss.read_index('/path/to/index.faiss').ntotal)"
   
   # Check metadata
   head -5 /path/to/metadata.json
   ```

### Debug Commands

```bash
# Check service status
curl http://localhost:5000/health

# Test audio processing
curl -X POST -F "audio=@audio.wav" http://localhost:5000/transcribe

# Test search functionality
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "test question"}' http://localhost:5000/search
```

## 🔄 Deployment

### Production Deployment

1. **Environment Setup**
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
export HF_TOKEN=your-hf-token
```

2. **Run with Gunicorn**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **Nginx Configuration**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Monitoring

- Health check endpoint: `/health`
- Service status monitoring
- Performance metrics collection
- Error tracking and alerting

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions
- Include error handling

### Testing

- Unit tests for all services
- Integration tests for API endpoints
- Performance benchmarks
- Error scenario testing

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AI4Bharat**: For IndicConformer and Cadence models
- **Facebook**: For MMS-LID language detection model
- **OpenAI**: For Whisper transcription model
- **Hugging Face**: For model hosting and transformers library
- **FAISS**: For efficient vector search capabilities

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation
- Contact the development team

---

**Note**: This backend service is designed to work with external transcription and translation APIs. Ensure these services are properly configured and accessible for full functionality.