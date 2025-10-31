#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multilingual Voice-Based Agricultural Recommendation System
Updated for TorchAudio 2.8+ deprecations and TorchCodec migration
Optimized for Hugging Face Spaces deployment with Whisper-first pipeline
"""

from __future__ import annotations
import torch
import warnings
import json
import os
import re
import tempfile
import shutil
import gradio as gr
import pandas as pd
from typing import List, Dict, Optional, Union
from transformers import AutoModel, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers import AutoModelForSeq2SeqLM
from pathlib import Path
import torch.nn as nn
from transformers import Gemma3ForCausalLM, Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer, 
    Gemma3TextModel,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import logging
from sentence_transformers import SentenceTransformer, util
import librosa  # Alternative to torchaudio
import soundfile as sf  # Alternative audio loading

# Try to import TorchCodec and TorchAudio with fallbacks
try:
    import torchcodec
    from torchcodec import AudioDecoder
    TORCHCODEC_AVAILABLE = True
    print("✅ TorchCodec available - using new audio loading")
except ImportError:
    TORCHCODEC_AVAILABLE = False
    print("⚠️ TorchCodec not available - using fallback methods")

try:
    import torchaudio
    # Suppress TorchAudio deprecation warnings for backends
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    TORCHAUDIO_AVAILABLE = True
    print("✅ TorchAudio available - with deprecation handling")
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None
    print("⚠️ TorchAudio not available - using librosa fallback")

try:
    from IndicTransToolkit.processor import IndicProcessor
    INDICTRANS_TOOLKIT_AVAILABLE = True
    print("✅ IndicTransToolkit available")
except ImportError:
    INDICTRANS_TOOLKIT_AVAILABLE = False
    print("⚠️ IndicTransToolkit not available - using basic preprocessing")

logger = logging.get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN", "")

if HF_TOKEN:
    from huggingface_hub import login
    try:
        login(HF_TOKEN)
        print("✅ Successfully logged in to Hugging Face!")
    except Exception as e:
        print(f"⚠️ HF login failed: {e}")

# --- FALLBACK INDIC PROCESSOR FOR WHEN TOOLKIT IS NOT AVAILABLE ---
class BasicIndicProcessor:
    """Basic fallback processor when IndicTransToolkit is not available"""
    def __init__(self, inference=True):
        self.inference = inference
    
    def preprocess_batch(self, sentences, src_lang, tgt_lang):
        """Basic preprocessing - add language tokens"""
        processed_sentences = []
        for sentence in sentences:
            processed_sentence = f"<2{tgt_lang}> {sentence.strip()}"
            processed_sentences.append(processed_sentence)
        return processed_sentences
    
    def postprocess_batch(self, sentences, lang):
        """Basic postprocessing - remove special tokens"""
        processed_sentences = []
        for sentence in sentences:
            processed_sentence = sentence.strip()
            if processed_sentence.startswith('<2'):
                processed_sentence = processed_sentence.split('>', 1)[-1].strip()
            processed_sentences.append(processed_sentence)
        return processed_sentences

# --- CUSTOM GEMMA3 BIDIRECTIONAL MODEL FOR PUNCTUATION ---
class Gemma3PunctuationConfig(Gemma3TextConfig):
    """Configuration class for Gemma3 punctuation model."""
    model_type = "cadence_punctuation"
    
    def __init__(
        self,
        num_labels: int = 31,
        classifier_dropout_prob: float = 0.0,
        use_non_causal_attention: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.classifier_dropout_prob = classifier_dropout_prob
        self.use_non_causal_attention = use_non_causal_attention

class NonCausalGemma3Attention(Gemma3Attention):
    """Gemma3Attention configured for non-causal token classification."""
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.sliding_window = None

class NonCausalGemma3DecoderLayer(Gemma3DecoderLayer):
    """Decoder layer with non-causal attention for token classification."""
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = NonCausalGemma3Attention(config, layer_idx)

class Gemma3TokenClassificationModel(Gemma3TextModel):
    """Gemma3 base model configured for token classification."""
    _no_split_modules = ["NonCausalGemma3DecoderLayer"]
    
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, 'use_non_causal_attention', True):
            self.layers = nn.ModuleList(
                [
                    NonCausalGemma3DecoderLayer(config, layer_idx)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values = None,
        output_attentions: bool = False,
    ):
        """Override to create bidirectional attention mask (no causal masking)."""
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
            
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, type(None)) is False and hasattr(past_key_values, 'get_max_length')
        
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
            
        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.zeros(
                (sequence_length, target_length), dtype=dtype, device=device
            )
            
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
                
        return causal_mask

class Gemma3ForTokenClassification(Gemma3ForCausalLM):
    """Gemma3 model for token classification (punctuation prediction)."""
    
    config_class = Gemma3PunctuationConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        if getattr(config, 'use_non_causal_attention', True):
            self.model = Gemma3TokenClassificationModel(config)
        
        classifier_dropout_prob = getattr(config, 'classifier_dropout_prob', 0.0)
        self.lm_head = nn.Sequential(
            nn.Dropout(classifier_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
        self.config.num_labels = config.num_labels
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> TokenClassifierOutput:
        """Forward pass for token classification."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Register the custom model
from transformers import AutoConfig
AutoConfig.register("cadence_punctuation", Gemma3PunctuationConfig)

# --- LANGUAGE MAPPINGS ---
LID_TO_ASR_LANG_MAP = {
    "asm_Beng": "as", "ben_Beng": "bn", "brx_Deva": "br", "doi_Deva": "doi",
    "guj_Gujr": "gu", "hin_Deva": "hi", "kan_Knda": "kn", "kas_Arab": "ks",
    "kas_Deva": "ks", "gom_Deva": "kok", "mai_Deva": "mai", "mal_Mlym": "ml",
    "mni_Beng": "mni", "mar_Deva": "mr", "nep_Deva": "ne", "ory_Orya": "or",
    "pan_Guru": "pa", "san_Deva": "sa", "sat_Olck": "sat", "snd_Arab": "sd",
    "tam_Taml": "ta", "tel_Telu": "te", "urd_Arab": "ur",
    "asm": "as", "ben": "bn", "brx": "br", "doi": "doi", "guj": "gu", "hin": "hi",
    "kan": "kn", "kas": "ks", "gom": "kok", "mai": "mai", "mal": "ml", "mni": "mni",
    "mar": "mr", "npi": "ne", "ory": "or", "pan": "pa", "sa": "sa", "sat": "sat",
    "snd": "sd", "tam": "ta", "tel": "te", "urd": "ur", "en": "en"
}

ASR_CODE_TO_NAME = {
    "as": "Assamese", "bn": "Bengali", "br": "Bodo", "doi": "Dogri", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "ks": "Kashmiri", "kok": "Konkani", "mai": "Maithili",
    "ml": "Malayalam", "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
    "pa": "Punjabi", "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu", "en": "English"
}

ASR_TO_INDICTRANS_MAP = {
    "as": "asm_Beng", "bn": "ben_Beng", "br": "brx_Deva", "doi": "doi_Deva",
    "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda", "ks": "kas_Deva",
    "kok": "gom_Deva", "mai": "mai_Deva", "ml": "mal_Mlym", "mni": "mni_Beng",
    "mr": "mar_Deva", "ne": "nep_Deva", "or": "ory_Orya", "pa": "pan_Guru",
    "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Arab", "ta": "tam_Taml",
    "te": "tel_Telu", "ur": "urd_Arab", "en": "eng_Latn"
}

# Audio processing configuration
SUPPORTED_AUDIO_FORMATS = {
    '.wav', '.mp3', '.flac', '.opus', '.ogg', '.m4a', '.aac', '.mp4',
    '.wma', '.amr', '.aiff', '.au', '.3gp', '.webm', '.mpeg'
}

def detect_audio_format(audio_path: str) -> str:
    return Path(audio_path).suffix.lower()

def load_audio_torchcodec(audio_path: str, target_sr: int = 16000) -> tuple:
    """Load audio using TorchCodec (new recommended method)"""
    try:
        print(f"🔧 Loading audio with TorchCodec: {audio_path}")
        
        # Use TorchCodec AudioDecoder
        decoder = AudioDecoder(audio_path)
        
        # Get audio info
        metadata = decoder.metadata
        original_sr = int(metadata.sample_rate)
        
        # Decode audio
        audio_data = decoder.decode()  # Returns tensor
        waveform = audio_data.audio  # Get audio tensor
        
        print(f"🎵 TorchCodec loaded audio: {waveform.shape} at {original_sr} Hz")
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print(f"🔄 Converted from stereo to mono")
        
        # Resample if needed
        if original_sr != target_sr:
            print(f"🔄 Resampling from {original_sr} Hz to {target_sr} Hz...")
            # Use torchaudio functional for resampling (still available)
            if TORCHAUDIO_AVAILABLE:
                waveform = torchaudio.functional.resample(
                    waveform,
                    orig_freq=original_sr,
                    new_freq=target_sr
                )
            else:
                # Fallback to librosa
                waveform_np = waveform.numpy()
                waveform_resampled = librosa.resample(
                    waveform_np[0], 
                    orig_sr=original_sr, 
                    target_sr=target_sr
                )
                waveform = torch.tensor(waveform_resampled).unsqueeze(0)
            print(f"✅ Resampled to {target_sr} Hz")
        
        print(f"✅ TorchCodec final audio: {waveform.shape} at {target_sr} Hz")
        return waveform, target_sr
        
    except Exception as e:
        print(f"❌ TorchCodec loading failed: {e}")
        raise e

def load_audio_librosa(audio_path: str, target_sr: int = 16000) -> tuple:
    """Load audio using librosa (fallback method)"""
    try:
        print(f"🔧 Loading audio with librosa: {audio_path}")
        
        # Load with librosa
        waveform_np, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Convert to torch tensor and add channel dimension
        waveform = torch.tensor(waveform_np).unsqueeze(0)
        
        print(f"✅ Librosa loaded audio: {waveform.shape} at {target_sr} Hz")
        return waveform, target_sr
        
    except Exception as e:
        print(f"❌ Librosa loading failed: {e}")
        raise e

def load_audio_torchaudio_legacy(audio_path: str, target_sr: int = 16000) -> tuple:
    """Load audio using legacy TorchAudio (with backend handling)"""
    try:
        print(f"🔧 Loading audio with TorchAudio (legacy): {audio_path}")
        
        # Try different backends
        backends_to_try = []
        
        if TORCHAUDIO_AVAILABLE:
            try:
                # Suppress the deprecation warning temporarily
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    available_backends = torchaudio.list_audio_backends()
                backends_to_try = available_backends
            except Exception:
                backends_to_try = ['soundfile', 'sox_io']
        
        audio_format = detect_audio_format(audio_path)
        print(f"🎵 Audio format: {audio_format}")
        print(f"🔧 Available backends: {backends_to_try}")
        
        waveform = None
        orig_sr = None
        
        # Try to load with different backends
        for backend in backends_to_try + [None]:  # None for default
            try:
                if backend:
                    print(f"🔄 Trying {backend} backend...")
                    if hasattr(torchaudio, 'set_audio_backend'):
                        torchaudio.set_audio_backend(backend)
                    waveform, orig_sr = torchaudio.load(audio_path, backend=backend)
                else:
                    print(f"🔄 Trying default backend...")
                    waveform, orig_sr = torchaudio.load(audio_path)
                
                print(f"✅ Successfully loaded with {backend or 'default'} backend")
                break
                
            except Exception as e:
                print(f"❌ {backend or 'default'} backend failed: {e}")
                continue
        
        if waveform is None:
            raise Exception("All TorchAudio backends failed")
        
        print(f"🎵 Loaded audio: {waveform.shape} at {orig_sr} Hz")
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print(f"🔄 Converted from stereo to mono")
        
        # Resample if needed
        if orig_sr != target_sr:
            print(f"🔄 Resampling from {orig_sr} Hz to {target_sr} Hz...")
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=orig_sr,
                new_freq=target_sr
            )
            print(f"✅ Resampled to {target_sr} Hz")
        
        return waveform, target_sr
        
    except Exception as e:
        print(f"❌ TorchAudio legacy loading failed: {e}")
        raise e

def preprocess_audio(audio_path: str, target_sr: int = 16000) -> tuple:
    """
    Preprocess audio with multiple fallback methods for TorchAudio 2.8+ compatibility
    """
    try:
        original_audio_format = detect_audio_format(audio_path)
        print(f"🎵 Detected original format: {original_audio_format}")
        
        # Method 1: Try TorchCodec (recommended for future)
        if TORCHCODEC_AVAILABLE:
            try:
                return load_audio_torchcodec(audio_path, target_sr)
            except Exception as e:
                print(f"⚠️ TorchCodec failed: {e}")
        
        # Method 2: Try TorchAudio legacy (with deprecation handling)
        if TORCHAUDIO_AVAILABLE:
            try:
                return load_audio_torchaudio_legacy(audio_path, target_sr)
            except Exception as e:
                print(f"⚠️ TorchAudio legacy failed: {e}")
        
        # Method 3: Fallback to librosa
        try:
            return load_audio_librosa(audio_path, target_sr)
        except Exception as e:
            print(f"⚠️ Librosa fallback failed: {e}")
        
        raise Exception("All audio loading methods failed")
        
    except Exception as e:
        error_msg = f"❌ Error in audio preprocessing: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

# --- GLOBAL MODEL STORAGE ---
models = {}
qa_system = {}

def load_models():
    """Load all models with caching using global variables."""
    global models
    
    if models:
        print("✅ Models already loaded from cache")
        return models
    
    print("🚀 Loading models for the first time...")
    
    try:
        print("Loading ASR model (IndicConformer)...")
        asr_model_id = "ai4bharat/indic-conformer-600m-multilingual"
        models['asr_model'] = AutoModel.from_pretrained(asr_model_id, trust_remote_code=True).to(device)
        models['asr_model'].eval()
        print("✅ ASR Model loaded.")
    except Exception as e:
        print(f"❌ Error loading ASR model: {e}")
        models['asr_model'] = None

    try:
        print("Loading Whisper model for English...")
        model_name = "openai/whisper-small"
        models['whisper_processor'] = WhisperProcessor.from_pretrained(model_name)
        models['whisper_model'] = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        print("✅ Whisper Model loaded.")
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        models['whisper_processor'] = None
        models['whisper_model'] = None

    try:
        print("Loading Language ID model (MMS-LID-1024)...")
        lid_model_id = "facebook/mms-lid-1024"
        models['lid_processor'] = Wav2Vec2FeatureExtractor.from_pretrained(lid_model_id)
        models['lid_model'] = AutoModelForAudioClassification.from_pretrained(lid_model_id).to(device)
        models['lid_model'].eval()
        print("✅ Language ID Model loaded.")
    except Exception as e:
        print(f"❌ Error loading LID model: {e}")
        models['lid_processor'] = None
        models['lid_model'] = None

    try:
        print("Loading Cadence punctuation model...")
        punctuation_model_name = "ai4bharat/Cadence"
        models['punctuation_tokenizer'] = AutoTokenizer.from_pretrained(punctuation_model_name)
        models['punctuation_model'] = Gemma3ForTokenClassification.from_pretrained(
            punctuation_model_name,
            trust_remote_code=True
        ).to(device)
        
        models['punctuation_id2label'] = {
            0: "O", 1: ".", 2: ",", 3: "?", 4: "-", 5: ";", 6: "_", 7: "!", 8: "'", 9: "...",
            10: "\"", 11: "।", 12: "(", 13: ")", 14: ":", 15: "؍", 16: "۔", 17: "؟",
            18: ".\"", 19: ").", 20: "),", 21: "\",", 22: "\".", 23: "?\"", 24: "\"?",
            25: "।\"", 26: "\"।", 27: "؍", 28: "᱾", 29: "॥", 30: "᱾।"
        }
        print(f"✅ Cadence Punctuation Model loaded")
    except Exception as e:
        print(f"❌ Error loading Cadence punctuation model: {e}")
        models['punctuation_tokenizer'] = None
        models['punctuation_model'] = None
        models['punctuation_id2label'] = None

    # Load IndicTrans2 model
    try:
        print("🔄 Loading IndicTrans2 for translation...")
        model_name = "ai4bharat/indictrans2-indic-en-1B"
        
        models['indictrans_tokenizer'] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        models['indictrans_model'] = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        
        # Use IndicTransToolkit if available, otherwise use basic processor
        if INDICTRANS_TOOLKIT_AVAILABLE:
            models['indic_processor'] = IndicProcessor(inference=True)
            print("✅ IndicTrans2 loaded with IndicTransToolkit")
        else:
            models['indic_processor'] = BasicIndicProcessor(inference=True)
            print("✅ IndicTrans2 loaded with basic processor")
            
    except Exception as e:
        print(f"❌ Error loading IndicTrans2 model: {e}")
        models['indictrans_tokenizer'] = None
        models['indictrans_model'] = None
        models['indic_processor'] = None

    return models

def load_qa_system():
    """Load Q&A system with caching using global variables."""
    global qa_system
    
    if qa_system:
        print("✅ Q&A system already loaded from cache")
        return qa_system
    
    print("🚀 Loading Q&A system for the first time...")
    
    try:
        if os.path.exists("cleaned_qa_dataset.xlsx"):
            df = pd.read_excel("cleaned_qa_dataset.xlsx")
            qa_pairs = df[['Question', 'Answer']].dropna().drop_duplicates().reset_index(drop=True)
            questions = qa_pairs['Question'].tolist()
            answers = qa_pairs['Answer'].tolist()
            
            print("Loading sentence transformer model...")
            model = SentenceTransformer('all-mpnet-base-v2')
            
            print("Generating embeddings for questions...")
            question_embeddings = model.encode(questions, convert_to_tensor=True)
            
            qa_system = {
                'model': model,
                'questions': questions,
                'answers': answers,
                'question_embeddings': question_embeddings
            }
            
            print(f"✅ Q&A system loaded with {len(questions)} questions")
            return qa_system
        else:
            print("⚠️ Q&A dataset not found. Please upload cleaned_qa_dataset.xlsx")
            return None
    except Exception as e:
        print(f"❌ Error loading Q&A system: {e}")
        return None

# --- PROCESSING FUNCTIONS ---
def add_punctuation(text):
    """Add punctuation using the custom bidirectional Gemma3 model"""
    if not text or not models.get('punctuation_model') or not models.get('punctuation_tokenizer'):
        return text

    try:
        inputs = models['punctuation_tokenizer'](
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs['input_ids'][0]

        with torch.no_grad():
            outputs = models['punctuation_model'](**inputs)
            predictions_for_sentence = torch.argmax(outputs.logits, dim=-1)[0]

        result_tokens_and_punctuation = []
        all_token_strings = models['punctuation_tokenizer'].convert_ids_to_tokens(input_ids.tolist())

        for i, token_id_value in enumerate(input_ids.tolist()):
            if inputs['attention_mask'][0][i] == 0:
                continue

            current_token_string = all_token_strings[i]
            is_special_token = token_id_value in models['punctuation_tokenizer'].all_special_ids

            if not is_special_token:
                result_tokens_and_punctuation.append(current_token_string)

            predicted_punctuation_id = predictions_for_sentence[i].item()
            punctuation_character = models['punctuation_id2label'].get(predicted_punctuation_id, "O")

            if punctuation_character != "O" and not is_special_token:
                result_tokens_and_punctuation.append(punctuation_character)

        punctuated_text = models['punctuation_tokenizer'].convert_tokens_to_string(result_tokens_and_punctuation)
        return punctuated_text

    except Exception as e:
        print(f"❌ Bidirectional punctuation failed: {e}")
        return text

def detect_language_with_whisper(audio_path):
    """Use Whisper to detect if audio is English or non-English"""
    try:
        if not models.get('whisper_model') or not models.get('whisper_processor'):
            return False, None
            
        print("🔍 Using Whisper for initial language detection...")
        
        waveform, sr = preprocess_audio(audio_path, target_sr=16000)
        
        input_features = models['whisper_processor'](
            waveform.squeeze(),
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features.to(device)
        
        with torch.no_grad():
            predicted_ids = models['whisper_model'].generate(
                input_features,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            transcription = models['whisper_processor'].batch_decode(
                predicted_ids.sequences, 
                skip_special_tokens=True
            )[0].strip()
            
            english_indicators = len(transcription) > 0 and any(
                word.lower() in transcription.lower() 
                for word in ['the', 'and', 'is', 'to', 'a', 'of', 'for', 'in', 'on', 'with', 'as', 'by']
            )
            
            ascii_ratio = sum(1 for c in transcription if ord(c) < 128) / max(len(transcription), 1)
            is_english = english_indicators and ascii_ratio > 0.7 and len(transcription.split()) > 1
            
            print(f"🎯 Whisper detection result: {'English' if is_english else 'Non-English'}")
            print(f"📝 Whisper transcription: '{transcription}'")
            
            return is_english, transcription if is_english else None
            
    except Exception as e:
        print(f"⚠️ Whisper language detection failed: {e}")
        return False, None

def translate_with_indictrans2(text: str, source_lang: str = "hin_Deva") -> Dict:
    """
    Translate Indic language text to English using IndicTrans2 model.
    """
    try:
        if not models.get('indictrans_model') or not models.get('indictrans_tokenizer') or not models.get('indic_processor'):
            return {
                "success": False,
                "error": "IndicTrans2 model not loaded",
                "translated_text": ""
            }
        
        print(f"🔄 Translating with IndicTrans2: {source_lang} -> eng_Latn")
        
        input_sentences = [text.strip()]
        
        # Preprocess with IndicProcessor
        batch = models['indic_processor'].preprocess_batch(
            input_sentences,
            src_lang=source_lang,
            tgt_lang="eng_Latn"
        )
        
        # Tokenize the sentences and generate input encodings
        inputs = models['indictrans_tokenizer'](
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        
        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = models['indictrans_model'].generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        
        # Decode the generated tokens into text
        generated_tokens = models['indictrans_tokenizer'].batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Postprocess the translations
        translations = models['indic_processor'].postprocess_batch(generated_tokens, lang="eng_Latn")
        
        translated_text = translations[0] if translations else ""
        
        return {
            "success": True,
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": "eng_Latn"
        }
        
    except Exception as e:
        print(f"❌ IndicTrans2 translation failed: {str(e)}")
        return {
            "success": False,
            "error": f"Translation error: {str(e)}",
            "translated_text": ""
        }

def semantic_qa_search(user_question, similarity_threshold=0.3, top_k=3):
    """Perform semantic search on Q&A dataset."""
    if not qa_system:
        return {
            "status": "error",
            "message": "Q&A system not available. Please upload the dataset."
        }
    
    try:
        user_question_embedding = qa_system['model'].encode(user_question, convert_to_tensor=True)
        similarities = util.cos_sim(user_question_embedding, qa_system['question_embeddings'])
        top_results = torch.topk(similarities, k=top_k)
        
        results = []
        for score, idx in zip(top_results.values[0], top_results.indices[0]):
            results.append({
                'similarity_score': score.item(),
                'question': qa_system['questions'][idx],
                'answer': qa_system['answers'][idx],
                'index': idx.item()
            })
        
        if results and results[0]['similarity_score'] >= similarity_threshold:
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "answer": result['answer'],
                    "matched_question": result['question'],
                    "similarity_score": result['similarity_score'],
                    "confidence": "High" if result['similarity_score'] > 0.7 else "Medium"
                })
            return {
                "status": "success",
                "results": formatted_results
            }
        else:
            formatted_suggestions = []
            for i, result in enumerate(results):
                formatted_suggestions.append({
                    "rank": i + 1,
                    "question": result['question'],
                    "similarity_score": result['similarity_score']
                })
            return {
                "status": "no_match",
                "message": "No highly relevant answer found in the dataset.",
                "suggestions": formatted_suggestions
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Semantic search failed: {str(e)}"
        }

def transcribe_audio_with_lid(audio_path):
    """Main transcription function with Whisper-first pipeline."""
    if not audio_path:
        return "Please provide an audio file.", "", ""

    try:
        waveform_16k, sr = preprocess_audio(audio_path, target_sr=16000)
    except Exception as e:
        return f"Error loading/preprocessing audio: {e}", "", ""

    try:
        # STEP 1: Use Whisper for initial language detection
        is_english, whisper_transcription = detect_language_with_whisper(audio_path)
        
        if is_english and whisper_transcription:
            # ENGLISH PIPELINE
            print("🇺🇸 Processing as English audio...")
            detected_lang_str = "Detected Language: English (Whisper Detection)"
            
            punctuated_transcription = add_punctuation(whisper_transcription)
            print(f"Original Whisper: {whisper_transcription}")
            print(f"With punctuation: {punctuated_transcription}")
            
            translation_result = punctuated_transcription
            
            return (
                detected_lang_str,
                punctuated_transcription,
                translation_result,
            )
        
        else:
            # NON-ENGLISH PIPELINE
            print("🌍 Processing as Non-English audio...")
            
            if not models.get('lid_model') or not models.get('lid_processor'):
                return "Language detection model not available.", "", ""
            
            print("🔍 Using MMS-LID for detailed language identification...")
            
            inputs = models['lid_processor'](waveform_16k.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = models['lid_model'](**inputs)

            logits = outputs[0]
            predicted_lid_id = logits.argmax(-1).item()
            detected_lid_code = models['lid_model'].config.id2label[predicted_lid_id]
            asr_lang_code = LID_TO_ASR_LANG_MAP.get(detected_lid_code)

            if not asr_lang_code:
                detected_lang_str = f"Detected '{detected_lid_code}', which is not supported by the ASR model."
                return detected_lang_str, "N/A", "N/A"

            detected_lang_name = ASR_CODE_TO_NAME.get(asr_lang_code, 'Unknown')
            detected_lang_str = f"Detected Language: {detected_lang_name} ({detected_lid_code})"
            print(detected_lang_str)

            if not models.get('asr_model'):
                return "ASR model not available.", "", ""

            print(f"🔤 Transcribing with IndicConformer ({detected_lang_name})...")
            with torch.no_grad():
                transcription = models['asr_model'](waveform_16k.to(device), asr_lang_code, "rnnt")
            print("✅ IndicConformer transcription complete.")

            punctuated_transcription = add_punctuation(transcription.strip()) if transcription else ""
            print(f"Original: {transcription}")
            print(f"With punctuation: {punctuated_transcription}")

            # Translation to English using IndicTrans2
            translation_result = ""
            translation_error = ""

            if punctuated_transcription:
                indictrans_lang_code = ASR_TO_INDICTRANS_MAP.get(asr_lang_code)
                if indictrans_lang_code:
                    print(f"🔄 Translating {detected_lang_name} to English with IndicTrans2...")
                    translation_response = translate_with_indictrans2(
                        punctuated_transcription,
                        indictrans_lang_code
                    )

                    if translation_response["success"]:
                        translation_result = translation_response["translated_text"]
                        print("✅ IndicTrans2 translation complete.")
                    else:
                        translation_error = translation_response["error"]
                        translation_result = "Translation failed"
                        print(f"❌ Translation failed: {translation_error}")
                else:
                    translation_result = "Translation not supported for this language"
                    print(translation_result)
            else:
                translation_result = "No text to translate"

            if translation_error:
                translation_display = f"❌ {translation_result}\nError: {translation_error}"
            else:
                translation_display = translation_result

            return (
                detected_lang_str,
                punctuated_transcription,
                translation_display,
            )

    except Exception as e:
        return f"Error during processing: {str(e)}", "", ""

def process_audio_and_search(audio_path):
    """Process audio and perform semantic search."""
    print(f"--- Processing audio file with Whisper-first pipeline: {audio_path} ---")
    
    detected_language, transcription, translated_text = transcribe_audio_with_lid(audio_path)
    
    if "Error" in detected_language:
        return {
            "status": "audio_processing_failed",
            "error": detected_language
        }

    print("\n--- Performing semantic search on translated text ---")
    semantic_search_result = semantic_qa_search(translated_text)

    return {
        "status": "success",
        "audio_processing": {
            "detected_language": detected_language,
            "transcription": transcription,
            "translated_text": translated_text
        },
        "semantic_search": semantic_search_result
    }

# --- GRADIO INTERFACE ---
def gradio_interface_fn(audio_path):
    """Gradio wrapper function."""
    if not audio_path:
        return "No audio file provided", "", "", "Please upload an audio file."
    
    integrated_result = process_audio_and_search(audio_path)

    detected_language_output = ""
    transcription_output = ""
    translated_text_output = ""
    semantic_search_output_string = ""

    if integrated_result["status"] == "success":
        audio_processing = integrated_result["audio_processing"]
        detected_language_output = audio_processing["detected_language"]
        transcription_output = audio_processing["transcription"]
        translated_text_output = audio_processing["translated_text"]

        semantic_search = integrated_result["semantic_search"]

        if semantic_search["status"] == "success":
            semantic_search_output_string = "--- Top 3 Semantic Search Results ---\n\n"
            for result in semantic_search["results"]:
                semantic_search_output_string += (
                    f"Rank {result['rank']} ({result['confidence']} Confidence, Score: {result['similarity_score']:.3f})\n"
                    f"Matched Question: {result['matched_question']}\n"
                    f"Answer: {result['answer']}\n\n"
                )
        else:
            semantic_search_output_string = f"--- Semantic Search ---\n\n❌ {semantic_search['message']}\n\n"
            if 'suggestions' in semantic_search:
                semantic_search_output_string += "🔍 Top Suggestions:\n"
                for suggestion in semantic_search["suggestions"]:
                    semantic_search_output_string += (
                        f"- {suggestion['question']} (Score: {suggestion['similarity_score']:.3f})\n"
                    )

    else:
        error_message = integrated_result.get("error", "An unknown error occurred during audio processing.")
        detected_language_output = f"Error: {error_message}"
        transcription_output = "N/A"
        translated_text_output = "N/A"
        semantic_search_output_string = "Semantic search could not be performed due to audio processing error."

    return (detected_language_output, transcription_output, translated_text_output, semantic_search_output_string)

def create_gradio_app():
    """Create the Gradio interface."""
    
    audio_input = gr.Audio(type="filepath", label="Upload Audio File")
    detected_language_output = gr.Textbox(label="Detected Language")
    transcription_output = gr.Textbox(label="Transcription")
    translated_text_output = gr.Textbox(label="Translated Text")
    semantic_search_output = gr.Textbox(label="Semantic Search Results")

    audio_backend_info = ""
    if TORCHCODEC_AVAILABLE:
        audio_backend_info = "🎵 **Audio Backend**: TorchCodec (recommended)"
    elif TORCHAUDIO_AVAILABLE:
        audio_backend_info = "🎵 **Audio Backend**: TorchAudio (legacy with deprecation handling)"
    else:
        audio_backend_info = "🎵 **Audio Backend**: Librosa (fallback)"

    iface = gr.Interface(
        fn=gradio_interface_fn,
        inputs=audio_input,
        outputs=[detected_language_output, transcription_output, translated_text_output, semantic_search_output],
        title="🌾 Multilingual Agricultural Voice Assistant",
        description=f"""
        Upload an audio file in English or any of the 22+ supported Indic languages. 
        The system will:
        1. 🎧 Detect the language automatically
        2. 📝 Transcribe the speech with punctuation
        3. 🌍 Translate to English using **IndicTrans2**
        4. 🔍 Find relevant agricultural answers from the knowledge base
        
        **Supported Languages:** English, Hindi, Bengali, Telugu, Tamil, Gujarati, Kannada, Malayalam, Marathi, Punjabi, Odia, Assamese, Urdu, Nepali, Sanskrit, and more!
        
        {audio_backend_info}
        **🔧 Translation**: IndicTrans2 with robust preprocessing
        **⚠️ Note**: Updated for TorchAudio 2.8+ deprecations
        """,
        examples=[],
        theme=gr.themes.Soft(),
        allow_flagging="never",
    )
    
    return iface

# --- MAIN APPLICATION ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌾 MULTILINGUAL AGRICULTURAL VOICE ASSISTANT")
    print("="*60)
    
    if TORCHCODEC_AVAILABLE:
        print("🎵 Audio Backend: TorchCodec (recommended)")
    elif TORCHAUDIO_AVAILABLE:
        print("🎵 Audio Backend: TorchAudio (legacy with deprecation handling)")
    else:
        print("🎵 Audio Backend: Librosa (fallback)")
    
    print("🔧 Translation: IndicTrans2 Model")
    print("⚠️ Updated for TorchAudio 2.8+ deprecations")
    print("🎯 Features available:")
    print("   • Multi-format audio processing (15+ formats)")
    print("   • Whisper-based English detection and transcription")
    print("   • MMS-LID for 22+ Indic language detection")
    print("   • IndicConformer for Indic language ASR")
    print("   • Bidirectional Gemma3 punctuation (31 punctuation types)")
    print("   • IndicTrans2 for professional translation")
    print("   • Semantic Q&A search")
    print("="*60)
    
    print("🚀 Loading models...")
    models = load_models()
    qa_system = load_qa_system()
    
    print("🎪 Launching Gradio interface...")
    app = create_gradio_app()
    app.launch()
