"""
Configuration file for Voice Recommendation System API
"""

import os
from pathlib import Path

class Config:
    """Base configuration."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'voice-rec-system-dev-key-2024'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    
    # File upload settings
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {
        'wav', 'mp3', 'flac', 'opus', 'ogg', 'm4a', 
        'aac', 'mp4', 'wma', 'amr', 'aiff', 'au', 
        '3gp', 'webm', 'mpeg', 'webm', 'weba', 'mka',
        'mkv', 'avi', 'mov', 'qt', 'wmv', 'asf'
    }
    
    # Model configurations
    SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
    ASR_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
    WHISPER_MODEL_NAME = "openai/whisper-small"
    LID_MODEL_ID = "facebook/mms-lid-126"
    
    # Hugging Face settings
    HF_TOKEN = os.environ.get('HF_TOKEN')
    
    # Data paths
    BASE_DIR = Path(__file__).parent.parent
    QA_DATASET_PATH = BASE_DIR / "embedding_data" / "all_crops" / "Cleaned_QA_Audio.xlsx"
    
    # Audio processing settings
    TARGET_SAMPLE_RATE = 16000
    
    # Semantic search settings
    DEFAULT_SIMILARITY_THRESHOLD = 0.3
    DEFAULT_TOP_K = 4
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    
    # Translation API settings (if using external translation service)
    TRANSLATION_API_BASE_URL = os.environ.get('TRANSLATION_API_BASE_URL')
    TRANSLATION_API_HEADERS = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }
    
    # Transcription API settings - dynamically detect if running through ngrok
    @staticmethod
    def get_transcription_api_url():
        """Dynamically determine transcription API URL based on environment."""
        # Check if we're running through ngrok tunnel
        if os.environ.get('NGROK_DOMAIN'):
            return f"https://{os.environ.get('NGROK_DOMAIN')}/api/transcription"
        
        # Check if we're running in a containerized environment with ngrok
        if os.environ.get('HOSTNAME') and 'ngrok' in os.environ.get('HOSTNAME', ''):
            ngrok_domain = os.environ.get('NGROK_DOMAIN')
            if ngrok_domain:
                return f"https://{ngrok_domain}/api/transcription"
        
        # Default to localhost for development
        return os.environ.get("TRANSCRIPTION_API_URL", "http://localhost:8000")
    
    @staticmethod
    def get_transcription_api_url_from_request(request_headers=None):
        """Dynamically determine transcription API URL based on request headers."""
        if request_headers:
            # Check if request is coming through ngrok
            host = request_headers.get('host', '')
            if 'ngrok-free.dev' in host or 'ngrok.io' in host:
                return f"https://{host}/api/transcription"
            
            # Check for ngrok headers
            if request_headers.get('ngrok-skip-browser-warning'):
                ngrok_domain = os.environ.get('NGROK_DOMAIN')
                if ngrok_domain:
                    return f"https://{ngrok_domain}/api/transcription"
        
        # Fallback to environment-based detection
        return Config.get_transcription_api_url()
    
    # Set the transcription API URL
    TRANSCRIPTION_API_URL = get_transcription_api_url()

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    # Override with more secure settings for production
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'voice-rec-system-dev-key-2024'
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for production environment")

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB for testing

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Language mappings
LANGUAGE_MAPPINGS = {
    'LID_TO_ASR_LANG_MAP': {
        "asm_Beng": "as", "ben_Beng": "bn", "brx_Deva": "br", "doi_Deva": "doi",
        "guj_Gujr": "gu", "hin_Deva": "hi", "kan_Knda": "kn", "kas_Arab": "ks",
        "kas_Deva": "ks", "gom_Deva": "kok", "mai_Deva": "mai", "mal_Mlym": "ml",
        "mni_Beng": "mni", "mar_Deva": "mr", "nep_Deva": "ne", "ory_Orya": "or",
        "pan_Guru": "pa", "san_Deva": "sa", "sat_Olck": "sat", "snd_Arab": "sd",
        "tam_Taml": "ta", "tel_Telu": "te", "urd_Arab": "ur",
        "asm": "as", "ben": "bn", "brx": "br", "doi": "doi", "guj": "gu", "hin": "hi",
        "kan": "kn", "kas": "ks", "gom": "kok", "mai": "mai", "mal": "ml", "mni": "mni",
        "mar": "mr", "npi": "ne", "ory": "or", "pan": "pa", "sa": "sa", "sat": "sat",
        "snd": "sd", "tam": "ta", "tel": "te", "urd": "ur", "eng": "eng"
    },
    
    'ASR_CODE_TO_NAME': {
        "as": "Assamese", "bn": "Bengali", "br": "Bodo", "doi": "Dogri", "gu": "Gujarati",
        "hi": "Hindi", "kn": "Kannada", "ks": "Kashmiri", "kok": "Konkani", "mai": "Maithili",
        "ml": "Malayalam", "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
        "pa": "Punjabi", "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
        "te": "Telugu", "ur": "Urdu", "eng": "English"
    },
    
    'ASR_TO_INDICTRANS_MAP': {
        "as": "asm_Beng", "bn": "ben_Beng", "br": "brx_Deva", "doi": "doi_Deva",
        "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda", "ks": "kas_Deva",
        "kok": "gom_Deva", "mai": "mai_Deva", "ml": "mal_Mlym", "mni": "mni_Beng",
        "mr": "mar_Deva", "ne": "nep_Deva", "or": "ory_Orya", "pa": "pan_Guru",
        "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Arab", "ta": "tam_Taml",
        "te": "tel_Telu", "ur": "urd_Arab", "eng": "eng_Latn"
    },
    
    'LANGUAGE_OPTIONS': {
        "English": "eng_Latn",
        "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Telugu": "tel_Telu",
        "Tamil": "tam_Taml", "Gujarati": "guj_Gujr", "Kannada": "kan_Knda",
        "Malayalam": "mal_Mlym", "Marathi": "mar_Deva", "Punjabi": "pan_Guru",
        "Odia": "ory_Orya", "Assamese": "asm_Beng", "Urdu": "urd_Arab",
        "Nepali": "nep_Deva", "Sanskrit": "san_Deva", "Kashmiri": "kas_Deva",
        "Sindhi": "snd_Arab", "Bodo": "brx_Deva", "Dogri": "doi_Deva",
        "Konkani": "gom_Deva", "Maithili": "mai_Deva", "Manipuri": "mni_Beng",
        "Santali": "sat_Olck"
    }
}
