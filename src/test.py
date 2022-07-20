
from transformers import BartTokenizer, BartModel
import torch
import os
tri_file = os.path.join('/home/simon/wikidata5m', "wikidata5m_transductive_train.txt")#'/home/simon/桌面/closed-book-prompt-qa/src/test.txt'# 
triple_list = {}
f = open(tri_file, "r")
for line in f.readlines():
    h, r, t = line.strip().split("\t")   
    if r in triple_list:
        triple_list[r].append((h, t))
    else:
        triple_list[r] = [(h, t)]
# print(triple_list.items())#dict_items([('P31', [('Q29387131', 'Q5'), ('Q14946683', 'Q5')]), ('P1412', [('Q326660', 'Q652')]), ('P57', [('Q7339549', 'Q1365729')]), ('P27', [('Q554335', 'Q29999'), ('Q4221140', 'Q399')]), ('P54', [('Q20641639', 'Q80955')]), ('P131', [('Q6925786', 'Q488653')]), ('P19', [('Q4890993', 'Q931116')]), ('P156', [('Q3198638', 'Q2859200')]), ('P161', [('Q24905727', 'Q88139')])])
import heapq
top_n = 20#825
# print(list(triple_list.items())[0][1])
n_top_rel = list(heapq.nlargest(top_n, list(triple_list.items()), key=lambda s: len(s[1])))
top_rel = [id for id, tlist in n_top_rel]
tri_per_rel =  [len(tlist) for id, tlist in n_top_rel][-1]
print(tri_per_rel)
with open('/home/simon/桌面/closed-book-prompt-qa/output/relation_triple.txt', "w", encoding="utf8") as writer:      
  for id, tlist in n_top_rel:
    writer.write(str(id)+'\t'+str(len(tlist))+'\n')

        

# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# model = BartModel.from_pretrained("facebook/bart-base")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# print(inputs)

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)#torch.Size([1, 8, 768])

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