
from transformers import BartTokenizer, BartModel
import torch

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartModel.from_pretrained("facebook/bart-base")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print(inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)#torch.Size([1, 8, 768])

'''
BartModel(
  (shared_parameters): ModuleDict()
  (invertible_adapters): ModuleDict()
  (shared): Embedding(50265, 768, padding_idx=1)
  (encoder): BartEncoder(
    (invertible_adapters): ModuleDict()
    (embed_tokens): Embedding(50265, 768, padding_idx=1)
    (embed_positions): BartLearnedPositionalEmbedding(1026, 768)
    (layers): ModuleList(
      (0): BartEncoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (1): BartEncoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (2): BartEncoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (3): BartEncoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (4): BartEncoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (5): BartEncoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
    )
    (layernorm_embedding): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (decoder): BartDecoder(
    (embed_tokens): Embedding(50265, 768, padding_idx=1)
    (embed_positions): BartLearnedPositionalEmbedding(1026, 768)
    (layers): ModuleList(
      (0): BartDecoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (activation_fn): GELUActivation()
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (cross_attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (1): BartDecoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (activation_fn): GELUActivation()
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (cross_attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (2): BartDecoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (activation_fn): GELUActivation()
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (cross_attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (3): BartDecoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (activation_fn): GELUActivation()
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (cross_attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (4): BartDecoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (activation_fn): GELUActivation()
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (cross_attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
      (5): BartDecoderLayer(
        (self_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (activation_fn): GELUActivation()
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): BartAttention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
          (prefix_tuning): PrefixTuningShim(
            (pool): PrefixTuningPool(
              (prefix_tunings): ModuleDict()
            )
          )
        )
        (encoder_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (output_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
        (cross_attention_adapters): AdapterLayer(
          (adapters): ModuleDict()
          (adapter_fusion_layer): ModuleDict()
        )
      )
    )
    (layernorm_embedding): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (prefix_tuning): PrefixTuningPool(
    (prefix_tunings): ModuleDict()
  )
)
'''