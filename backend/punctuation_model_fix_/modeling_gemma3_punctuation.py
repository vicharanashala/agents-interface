"""
Change the attention of Gemma3 to be bidirectional.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from functools import partial

from transformers import PretrainedConfig, PreTrainedModel
from transformers import Gemma3ForCausalLM, Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer, 
    Gemma3TextModel,

)

from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Gemma3PunctuationConfig(Gemma3TextConfig):
    """
    Configuration class for Gemma3 punctuation model.
    """
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


# ============ Token Classification Model Components ============

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
            # Replace layers with non-causal versions
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
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            # KEY CHANGE: Start with zeros (attend to all) instead of min_dtype (mask all)
            causal_mask = torch.zeros(
                (sequence_length, target_length), dtype=dtype, device=device
            )
            # REMOVED: Causal masking lines that would make it lower triangular
            # if sequence_length != 1:
            #     causal_mask = torch.triu(causal_mask, diagonal=1)
            
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        # Handle SDPA-specific optimizations if needed
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            try:
                from transformers.modeling_attn_mask_utils import AttentionMaskConverter
                causal_mask = AttentionMaskConverter._unmask_unattended(
                    causal_mask, min_dtype
                )
            except ImportError:
                pass  # Fallback for older transformers versions

        return causal_mask


class Gemma3ForTokenClassification(Gemma3ForCausalLM):
    """
    Gemma3 model for token classification (punctuation prediction).
    Uses class-based architecture without monkey patching.
    """
    
    config_class = Gemma3PunctuationConfig
    
    def __init__(self, config):
        # Initialize with base Gemma3ForCausalLM structure
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Replace the base model with token classification version
        if getattr(config, 'use_non_causal_attention', True):
            self.model = Gemma3TokenClassificationModel(config)
        
        # Replace the lm_head with classification head
        classifier_dropout_prob = getattr(config, 'classifier_dropout_prob', 0.0)
        self.lm_head = nn.Sequential(
            nn.Dropout(classifier_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
        # Update config for classification
        self.config.num_labels = config.num_labels
        
        # Initialize weights for the new head
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

        # Get hidden states from the model
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

        # Get the hidden states from the model output
        sequence_output = outputs[0]
        
        # Apply the classification head (which is now self.lm_head)
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


# ============ Model Registration ============

from transformers import AutoConfig, AutoModel

# Register the punctuation config and model
AutoConfig.register("cadence_punctuation", Gemma3PunctuationConfig)
AutoModel.register(Gemma3PunctuationConfig, Gemma3ForTokenClassification)


# ============ Utility Functions ============


def create_token_classification_model(config: Gemma3PunctuationConfig):
    """Create a token classification model with non-causal attention."""
    return Gemma3ForTokenClassification(config)


def load_from_pretrained_with_config_detection(model_path: str, **kwargs):
    """
    Load model and auto-detect whether it's for token classification or bidirectional tasks
    based on the config.
    """
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_path)
    
    if hasattr(config, 'model_type') and config.model_type == "cadence_punctuation":
        # Token classification model
        return Gemma3ForTokenClassification.from_pretrained(model_path, config=config, **kwargs)
