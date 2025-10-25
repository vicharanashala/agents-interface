"""
Hybrid Language Detection Service
Combines Whisper and Facebook MMS-LID models for optimal language detection.
"""

import os
import logging
import torch
import whisper
from typing import Dict, Tuple, Optional
from transformers import (
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor
)

logger = logging.getLogger(__name__)

class HybridLanguageDetection:
    """Hybrid language detection using Whisper + Facebook MMS-LID fallback."""
    
    def __init__(self, config, models=None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = models or {}
        
        # Indian language codes and their mappings
        self.indian_languages = {
            'as': 'Assamese',
            'bn': 'Bengali', 
            'gu': 'Gujarati',
            'hi': 'Hindi',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'ne': 'Nepali',
            'or': 'Odia',
            'pa': 'Punjabi',
            'sa': 'Sanskrit',
            'sd': 'Sindhi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ur': 'Urdu'
        }
        
        # Confidence threshold for Indian languages
        self.confidence_threshold = 0.80
        
    def initialize_models(self) -> bool:
        """Initialize Whisper and MMS-LID models."""
        try:
            # Check if MMS-LID model is already provided from AudioService
            if self.models and 'lid_model' in self.models and 'lid_processor' in self.models:
                logger.info("Using pre-loaded MMS-LID model for hybrid language detection")
                # Map the shared MMS-LID model to the expected keys for this service
                self.models['mms_lid_processor'] = self.models['lid_processor']
                self.models['mms_lid_model'] = self.models['lid_model']
            else:
                # Load MMS-LID 126 model if not provided
                logger.info("Loading Facebook MMS-LID 126 model...")
                mms_lid_model_id = getattr(self.config, 'LID_MODEL_ID', 'facebook/mms-lid-126')
                self.models['mms_lid_processor'] = Wav2Vec2FeatureExtractor.from_pretrained(mms_lid_model_id)
                self.models['mms_lid_model'] = AutoModelForAudioClassification.from_pretrained(mms_lid_model_id).to(self.device)
                self.models['mms_lid_model'].eval()
            
            # Always load Whisper model for language detection (different from transcription)
            logger.info("Loading Whisper model for language detection...")
            self.models['whisper'] = whisper.load_model("small")
            
            logger.info("Hybrid language detection models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading hybrid language detection models: {str(e)}")
            return False
    
    def detect_language_with_whisper(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect language using Whisper model.
        Returns (language_code, confidence_score)
        """
        try:
            # Load and preprocess audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Convert to log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.models['whisper'].device)
            
            # Detect language
            _, probs = self.models['whisper'].detect_language(mel)
            
            # Get the language with highest probability
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            logger.debug(f"Whisper detected: {detected_lang} with confidence {confidence:.3f}")
            
            return detected_lang, confidence
            
        except Exception as e:
            logger.error(f"Error in Whisper language detection: {str(e)}")
            return None, 0.0
    
    def detect_language_with_mms_lid(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect language using Facebook MMS-LID 126 model.
        Returns (language_code, confidence_score)
        """
        try:
            # Load audio using whisper (for consistency)
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Convert to tensor and move to device
            audio_tensor = torch.from_numpy(audio).to(self.device)
            
            # Process with MMS-LID
            inputs = self.models['mms_lid_processor'](
                audio_tensor,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['mms_lid_model'](**inputs)
                logits = outputs.logits
                
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
            
            # Get language code from model config
            detected_lang = self.models['mms_lid_model'].config.id2label[predicted_id]
            
            logger.debug(f"MMS-LID detected: {detected_lang} with confidence {confidence:.3f}")
            
            return detected_lang, confidence
            
        except Exception as e:
            logger.error(f"Error in MMS-LID language detection: {str(e)}")
            return None, 0.0
    
    def detect_language_hybrid(self, audio_path: str) -> Dict:
        """
        Hybrid language detection:
        1. Use Whisper first
        2. If English detected, return English
        3. If Indian language detected with confidence > 80%, return that language
        4. Otherwise, use MMS-LID as fallback
        """
        try:
            logger.info(f"Starting hybrid language detection for: {audio_path}")
            
            # Step 1: Try Whisper first
            whisper_lang, whisper_conf = self.detect_language_with_whisper(audio_path)
            
            if whisper_lang is None:
                logger.warning("Whisper detection failed, falling back to MMS-LID")
                return self._fallback_to_mms_lid(audio_path)
            
            # Step 2: If English detected, return English
            if whisper_lang == 'en':
                logger.info(f"Whisper detected English with confidence {whisper_conf:.3f}")
                return {
                    'success': True,
                    'detected_language': 'English',
                    'language_code': 'en',
                    'confidence': whisper_conf,
                    'detection_method': 'whisper',
                    'model_used': 'whisper-small'
                }
            
            # Step 3: Check if it's an Indian language with high confidence
            if whisper_lang in self.indian_languages and whisper_conf >= self.confidence_threshold:
                lang_name = self.indian_languages[whisper_lang]
                logger.info(f"Whisper detected {lang_name} with high confidence {whisper_conf:.3f}")
                return {
                    'success': True,
                    'detected_language': lang_name,
                    'language_code': whisper_lang,
                    'confidence': whisper_conf,
                    'detection_method': 'whisper',
                    'model_used': 'whisper-small'
                }
            
            # Step 4: Low confidence or non-Indian language, fallback to MMS-LID
            logger.info(f"Whisper detected {whisper_lang} with low confidence {whisper_conf:.3f}, using MMS-LID fallback")
            return self._fallback_to_mms_lid(audio_path, whisper_result=(whisper_lang, whisper_conf))
            
        except Exception as e:
            logger.error(f"Error in hybrid language detection: {str(e)}")
            return {
                'success': False,
                'error': f"Hybrid language detection failed: {str(e)}",
                'detection_method': 'error'
            }
    
    # Alias for backward compatibility
    def detect_language(self, audio_path: str) -> Dict:
        """Alias for detect_language_hybrid for backward compatibility."""
        return self.detect_language_hybrid(audio_path)
    
    def _fallback_to_mms_lid(self, audio_path: str, whisper_result: Optional[Tuple[str, float]] = None) -> Dict:
        """
        Fallback to MMS-LID 126 model for language detection.
        """
        try:
            logger.info("Using MMS-LID 126 model for language detection")
            
            mms_lang, mms_conf = self.detect_language_with_mms_lid(audio_path)
            
            if mms_lang is None:
                return {
                    'success': False,
                    'error': "Both Whisper and MMS-LID language detection failed",
                    'detection_method': 'fallback_failed',
                    'whisper_result': whisper_result
                }
            
            # Map MMS-LID result to our language codes
            mapped_lang_code = self._map_mms_lid_to_language_code(mms_lang)
            lang_name = self.indian_languages.get(mapped_lang_code, mms_lang)
            
            # Handle English specifically
            if mapped_lang_code == 'eng' or mms_lang.lower() == 'eng':
                lang_name = 'English'
                mapped_lang_code = 'en'
            
            result = {
                'success': True,
                'detected_language': lang_name,
                'language_code': mapped_lang_code,
                'confidence': mms_conf,
                'detection_method': 'mms_lid_fallback',
                'model_used': 'facebook/mms-lid-126',
                'mms_lid_raw': mms_lang
            }
            
            if whisper_result:
                result['whisper_result'] = {
                    'language': whisper_result[0],
                    'confidence': whisper_result[1]
                }
            
            logger.info(f"MMS-LID detected: {lang_name} ({mapped_lang_code}) with confidence {mms_conf:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in MMS-LID fallback: {str(e)}")
            return {
                'success': False,
                'error': f"MMS-LID fallback failed: {str(e)}",
                'detection_method': 'mms_lid_error',
                'whisper_result': whisper_result
            }
    
    def _map_mms_lid_to_language_code(self, mms_lang: str) -> str:
        """
        Map MMS-LID language codes to our standard language codes.
        """
        # Common mappings from MMS-LID to our codes
        mapping = {
            'eng': 'en',
            'hin': 'hi',
            'ben': 'bn',
            'tel': 'te',
            'tam': 'ta',
            'guj': 'gu',
            'kan': 'kn',
            'mal': 'ml',
            'mar': 'mr',
            'pan': 'pa',
            'ori': 'or',
            'asm': 'as',
            'nep': 'ne',
            'san': 'sa',
            'snd': 'sd',
            'urd': 'ur'
        }
        
        # Try direct mapping first
        if mms_lang in mapping:
            return mapping[mms_lang]
        
        # Try case-insensitive mapping
        mms_lang_lower = mms_lang.lower()
        for key, value in mapping.items():
            if key.lower() == mms_lang_lower:
                return value
        
        # If no mapping found, return the original code
        logger.warning(f"No mapping found for MMS-LID language: {mms_lang}")
        return mms_lang
    
    def get_supported_languages(self) -> Dict:
        """Get list of supported languages."""
        return {
            'indian_languages': self.indian_languages,
            'confidence_threshold': self.confidence_threshold,
            'supported_codes': list(self.indian_languages.keys()) + ['en']
        }
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate if the audio file exists and is accessible."""
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"Audio file does not exist: {audio_path}")
            return False
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            return False
        
        return True


# Backward compatibility alias
MmsLidLanguageDetection = HybridLanguageDetection