"""
Punctuation service for voice recommendation system.
Handles text punctuation using the Cadence model.
"""

import logging
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class PunctuationService:
    """Service class for text punctuation operations."""
    
    def __init__(self, config):
        self.config = config
        # Use GPU for faster punctuation processing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.id2label = None
        
    def initialize_model(self) -> bool:
        """Initialize the punctuation model."""
        try:
            logger.info("Loading punctuation model (Cadence)...")
            model_name = getattr(self.config, 'PUNCTUATION_MODEL_ID', 'ai4bharat/Cadence')
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Get label mapping
            self.id2label = self.model.config.id2label
            
            logger.info("Punctuation model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading punctuation model: {str(e)}")
            return False
    
    def punctuate_text(self, text: str) -> Dict:
        """
        Add punctuation to the given text.
        
        Args:
            text (str): Input text without punctuation
            
        Returns:
            Dict: Result containing punctuated text and metadata
        """
        try:
            if not self.model or not self.tokenizer:
                return {
                    'success': False,
                    'error': 'Punctuation model not initialized'
                }
            
            if not text or not text.strip():
                return {
                    'success': False,
                    'error': 'Empty text provided'
                }
            
            # Tokenize input and prepare for model
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_ids = inputs['input_ids'][0]  # Get input_ids for the first (and only) sentence
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions_for_sentence = torch.argmax(outputs.logits, dim=-1)[0]
            
            # Process tokens and add punctuation
            result_tokens_and_punctuation = []
            all_token_strings = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
            
            for i, token_id_value in enumerate(input_ids.tolist()):
                # Process only non-padding tokens based on the attention mask
                if inputs['attention_mask'][0][i] == 0:
                    continue
                
                current_token_string = all_token_strings[i]
                is_special_token = token_id_value in self.tokenizer.all_special_ids
                
                if not is_special_token:
                    result_tokens_and_punctuation.append(current_token_string)
                
                predicted_punctuation_id = predictions_for_sentence[i].item()
                punctuation_character = self.id2label[predicted_punctuation_id]
                
                if punctuation_character != "O" and not is_special_token:
                    result_tokens_and_punctuation.append(punctuation_character)
            
            # Convert tokens back to string
            punctuated_text = self.tokenizer.convert_tokens_to_string(result_tokens_and_punctuation)
            
            return {
                'success': True,
                'original_text': text,
                'punctuated_text': punctuated_text,
                'character_count': len(punctuated_text),
                'word_count': len(punctuated_text.split()),
                'punctuation_added': len(punctuated_text) - len(text)
            }
            
        except Exception as e:
            logger.error(f"Error during punctuation: {str(e)}")
            return {
                'success': False,
                'error': f"Error during punctuation: {str(e)}",
                'original_text': text,
                'punctuated_text': text  # Return original text as fallback
            }
    
    def is_initialized(self) -> bool:
        """Check if the punctuation service is initialized."""
        return self.model is not None and self.tokenizer is not None
