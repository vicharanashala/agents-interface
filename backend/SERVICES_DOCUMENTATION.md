# Voice Recommendation System - Services Documentation

## Overview

This document provides detailed technical documentation for the services layer of the Voice Recommendation System backend. The services are modular components that handle specific functionalities like language detection, punctuation, and semantic search.

## Service Architecture

```
services/
├── __init__.py                    # Services module initialization
├── mms_lid_language_detection.py # Hybrid language detection
├── punctuation_service.py        # Text punctuation
└── search_service.py             # Semantic search
```

## 1. MmsLidLanguageDetection Service

### Purpose
Hybrid language detection service that combines Whisper and Facebook MMS-LID models for optimal language detection across Indian languages and English.

### Class: `HybridLanguageDetection`

#### Initialization
```python
detector = HybridLanguageDetection(config, models=None)
```

**Parameters**:
- `config`: Configuration object with model settings
- `models`: Optional pre-loaded models dictionary

#### Key Methods

##### `initialize_models() -> bool`
Initializes Whisper and MMS-LID models for language detection.

**Process**:
1. Loads Facebook MMS-LID 126 model
2. Loads Whisper small model for language detection
3. Sets up device (CUDA/CPU) configuration
4. Validates model loading

**Returns**: `True` if successful, `False` otherwise

##### `detect_language_hybrid(audio_path: str) -> Dict`
Main language detection method using hybrid approach.

**Algorithm**:
1. **Whisper Detection**: Use Whisper model first
2. **English Check**: If English detected, return immediately
3. **Confidence Check**: If Indian language with >80% confidence, return
4. **MMS-LID Fallback**: Use MMS-LID for low confidence or unknown languages

**Response Format**:
```python
{
    'success': True,
    'detected_language': 'Hindi',
    'language_code': 'hi',
    'confidence': 0.95,
    'detection_method': 'whisper|mms_lid_fallback',
    'model_used': 'whisper-small|facebook/mms-lid-126',
    'whisper_result': {  # Optional
        'language': 'hi',
        'confidence': 0.85
    }
}
```

##### `detect_language_with_whisper(audio_path: str) -> Tuple[str, float]`
Language detection using Whisper model.

**Process**:
1. Load audio using `whisper.load_audio()`
2. Convert to log-Mel spectrogram
3. Use Whisper's `detect_language()` method
4. Return language code and confidence

##### `detect_language_with_mms_lid(audio_path: str) -> Tuple[str, float]`
Language detection using Facebook MMS-LID model.

**Process**:
1. Load and preprocess audio
2. Extract features using Wav2Vec2FeatureExtractor
3. Run inference on MMS-LID model
4. Map predicted ID to language code

#### Supported Languages

**Indian Languages**:
- Assamese (as), Bengali (bn), Gujarati (gu), Hindi (hi)
- Kannada (kn), Malayalam (ml), Marathi (mr), Nepali (ne)
- Odia (or), Punjabi (pa), Sanskrit (sa), Sindhi (sd)
- Tamil (ta), Telugu (te), Urdu (ur)

**International**:
- English (en)

#### Configuration

```python
# Model Configuration
LID_MODEL_ID = "facebook/mms-lid-126"
WHISPER_MODEL = "small"

# Detection Parameters
confidence_threshold = 0.80  # For Indian languages
```

#### Error Handling

- **Model Loading Failures**: Graceful degradation with fallback
- **Audio Processing Errors**: Detailed error messages
- **Detection Failures**: Fallback to alternative methods

## 2. PunctuationService

### Purpose
Add punctuation to transcribed text using AI4Bharat's Cadence model, supporting multiple Indian languages.

### Class: `PunctuationService`

#### Initialization
```python
punctuation_service = PunctuationService(config)
```

**Parameters**:
- `config`: Configuration object with model settings

#### Key Methods

##### `initialize_model() -> bool`
Initialize the Cadence punctuation model.

**Process**:
1. Load tokenizer from model repository
2. Load Cadence model with `trust_remote_code=True`
3. Move model to appropriate device (CUDA/CPU)
4. Set model to evaluation mode
5. Extract label mapping from model config

**Model**: `ai4bharat/Cadence`

##### `punctuate_text(text: str) -> Dict`
Add punctuation to input text.

**Process**:
1. **Tokenization**: Convert text to tokens
2. **Model Inference**: Run through Cadence model
3. **Prediction**: Get punctuation predictions for each token
4. **Post-processing**: Combine tokens with punctuation
5. **Result Formatting**: Return structured response

**Response Format**:
```python
{
    'success': True,
    'original_text': 'मेरा नाम राम है',
    'punctuated_text': 'मेरा नाम राम है।',
    'character_count': 16,
    'word_count': 3,
    'punctuation_added': 1
}
```

**Error Handling**:
```python
{
    'success': False,
    'error': 'Detailed error message',
    'original_text': text,
    'punctuated_text': text  # Fallback to original
}
```

##### `is_initialized() -> bool`
Check if the punctuation service is properly initialized.

**Returns**: `True` if both model and tokenizer are loaded

#### Technical Details

**Model Architecture**:
- Based on transformer architecture
- Token-level punctuation prediction
- Supports multiple Indian languages
- GPU acceleration support

**Token Processing**:
1. Special tokens are preserved
2. Punctuation is added after non-special tokens
3. Label mapping: `id2label` from model config
4. "O" label indicates no punctuation

**Performance Optimizations**:
- GPU acceleration when available
- Batch processing support
- Efficient tokenization
- Memory management

#### Configuration

```python
# Model Configuration
PUNCTUATION_MODEL_ID = 'ai4bharat/Cadence'

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 3. SearchService

### Purpose
Semantic search service using FAISS index and sentence transformers for Q&A recommendation system.

### Class: `SearchService`

#### Initialization
```python
search_service = SearchService(config)
```

**Parameters**:
- `config`: Configuration object with model and path settings

#### Key Methods

##### `initialize_model() -> bool`
Initialize sentence transformer model for semantic search.

**Model**: `all-MiniLM-L6-v2` (default)
**Process**:
1. Load SentenceTransformer model
2. Configure for semantic search
3. Validate model loading

##### `load_qa_dataset() -> bool`
Load FAISS index and metadata for Q&A search.

**Files**:
- FAISS Index: `/workspace/voice_recommendation_system/notebook/embedding/paddy/complete_qa_index.faiss`
- Metadata: `/workspace/voice_recommendation_system/notebook/embedding/paddy/metadata.json`

**Process**:
1. Load FAISS index using `faiss.read_index()`
2. Load metadata from JSON lines format
3. Validate index and metadata compatibility
4. Verify data integrity

##### `semantic_search(user_question: str, similarity_threshold: float = None, top_k: int = None) -> Dict`
Perform semantic search using FAISS index.

**Algorithm**:
1. **Query Encoding**: Convert question to embedding using sentence transformer
2. **FAISS Search**: Search index for similar vectors
3. **Result Processing**: Format and rank results
4. **Similarity Scoring**: Convert distances to similarity scores

**Response Format**:
```python
{
    "status": "success",
    "results": [
        {
            "rank": 1,
            "answer": "Paddy cultivation involves...",
            "matched_question": "What is paddy cultivation?",
            "distance": 0.15,
            "similarity_score": 0.87,
            "confidence": "High"
        }
    ],
    "query": "What is paddy cultivation?",
    "total_results": 5
}
```

##### `search_by_keywords(keywords: List[str], top_k: int = 5) -> Dict`
Keyword-based search in Q&A dataset.

**Process**:
1. **Keyword Matching**: Count keyword occurrences in questions
2. **Ranking**: Sort by number of matches
3. **Filtering**: Return top-k results

**Response Format**:
```python
{
    'status': 'success',
    'results': [
        {
            'question': 'What is paddy cultivation?',
            'answer': 'Paddy cultivation involves...',
            'keyword_matches': 3,
            'index': 42
        }
    ],
    'total_matches': 15,
    'keywords_searched': ['paddy', 'cultivation']
}
```

##### `get_dataset_info() -> Dict`
Get information about the loaded dataset.

**Returns**:
```python
{
    'loaded': True,
    'total_pairs': 1000,
    'faiss_index_size': 1000,
    'avg_question_length': 15.5,
    'avg_answer_length': 45.2,
    'sample_questions': ['What is paddy?', 'How to grow rice?']
}
```

#### Technical Implementation

**FAISS Integration**:
- Uses FAISS for efficient vector search
- Supports various distance metrics
- Handles large-scale vector databases
- Optimized for similarity search

**Sentence Transformers**:
- Model: `all-MiniLM-L6-v2`
- Embedding dimension: 384
- Optimized for semantic similarity
- Fast inference capabilities

**Search Pipeline**:
1. **Query Processing**: Text preprocessing and normalization
2. **Embedding Generation**: Convert text to vector representation
3. **Vector Search**: FAISS-based similarity search
4. **Result Ranking**: Distance-based ranking with confidence scoring
5. **Response Formatting**: Structured JSON response

#### Configuration

```python
# Model Configuration
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

# Search Parameters
DEFAULT_TOP_K = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.7

# File Paths
faiss_index_path = '/workspace/voice_recommendation_system/notebook/embedding/paddy/complete_qa_index.faiss'
metadata_path = '/workspace/voice_recommendation_system/notebook/embedding/paddy/metadata.json'
```


## Service Integration

### Initialization Flow

```python
# 1. Load configuration
config = DevelopmentConfig()

# 2. Initialize services
mms_lid_detector = MmsLidLanguageDetection(config)
punctuation_service = PunctuationService(config)
search_service = SearchService(config)

# 3. Initialize models
mms_lid_detector.initialize_models()
punctuation_service.initialize_model()
search_service.initialize_model()
search_service.load_qa_dataset()
```

### Error Handling Strategy

**Graceful Degradation**:
- Services continue to operate if individual components fail
- Fallback mechanisms for critical operations
- Detailed error logging for debugging

**Service Status Tracking**:
- Health check endpoints report service status
- Initialization failures are logged but don't crash the system
- Optional services can be disabled without affecting core functionality

### Performance Considerations

**Model Loading**:
- Models are loaded once during startup
- GPU acceleration when available
- Memory management for large models

**Caching**:
- FAISS index is loaded once and cached
- Sentence transformer model is cached
- Metadata is loaded once and stored in memory

**Optimization**:
- Batch processing where possible
- Efficient tensor operations
- Memory-efficient data structures

## Configuration Management

### Service-Specific Configuration

```python
# Language Detection
LID_MODEL_ID = "facebook/mms-lid-126"
WHISPER_MODEL = "small"
confidence_threshold = 0.80

# Punctuation
PUNCTUATION_MODEL_ID = 'ai4bharat/Cadence'

# Search
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_TOP_K = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.3

# File Paths
faiss_index_path = '/path/to/index.faiss'
metadata_path = '/path/to/metadata.json'
```

### Environment Variables

```bash
# Model Configuration
HF_TOKEN=hf_your_token_here
CUDA_VISIBLE_DEVICES=0

# File Paths
FAISS_INDEX_PATH=/path/to/index.faiss
METADATA_PATH=/path/to/metadata.json

# Performance
MAX_WORKERS=4
BATCH_SIZE=32
```

## Testing and Validation

### Unit Testing

Each service should have comprehensive unit tests covering:
- Model initialization
- Core functionality
- Error handling
- Edge cases

### Integration Testing

- Service interaction testing
- End-to-end workflow validation
- Performance benchmarking
- Memory usage monitoring

### Validation Methods

```python
# Service Status Validation
def validate_services():
    services = {
        'mms_lid': mms_lid_detector.is_initialized(),
        'punctuation': punctuation_service.is_initialized(),
        'search': search_service.faiss_index is not None
    }
    return services

# Performance Monitoring
def benchmark_service(service, input_data):
    start_time = time.time()
    result = service.process(input_data)
    processing_time = time.time() - start_time
    return result, processing_time
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check Hugging Face token
   - Verify internet connectivity
   - Check disk space for model downloads

2. **FAISS Index Issues**
   - Verify index file exists and is readable
   - Check metadata file format
   - Ensure index and metadata compatibility

3. **GPU Memory Issues**
   - Monitor GPU memory usage
   - Implement model offloading
   - Use CPU fallback when needed

### Debug Tools

```python
# Service Status Check
def check_service_status():
    return {
        'mms_lid_loaded': mms_lid_detector.models is not None,
        'punctuation_loaded': punctuation_service.model is not None,
        'search_loaded': search_service.faiss_index is not None
    }

# Performance Profiling
def profile_service_performance():
    # Implementation for performance monitoring
    pass
```

## Future Enhancements

### Planned Improvements

1. **Model Optimization**
   - Quantization for faster inference
   - Model pruning for reduced memory usage
   - Custom fine-tuning capabilities

2. **Service Scalability**
   - Horizontal scaling support
   - Load balancing integration
   - Caching layer implementation

3. **Advanced Features**
   - Multi-modal search capabilities
   - Real-time model updates
   - Custom model integration

### Integration Opportunities

1. **External Services**
   - Cloud-based model hosting
   - Distributed inference systems
   - API gateway integration

2. **Monitoring and Analytics**
   - Service performance metrics
   - Usage analytics
   - Error tracking and alerting

---

This documentation provides comprehensive technical details for each service in the Voice Recommendation System. For implementation specifics, refer to the source code in the respective service files.
