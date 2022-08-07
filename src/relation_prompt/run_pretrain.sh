MODEL="facebook/bart-base"
TOKENIZER="facebook/bart-base"
# MODEL="roberta-base"
# TOKENIZER="roberta-base"
INPUT_DIR="/home/simon/wikidata5m"
OUTPUT_DIR="checkpoints"
DATASET_NAME="wikidata5m"
ADAPTER_NAMES="entity_predict"
ADAPTER_TYPE="PfeifferConfig" 
# ADAPTER_TYPE2="ParallelConfig"
# ADAPTER_TYPE4="PrefixTuningConfig"
# ADAPTER_TYPE5="CompacterConfig"

# ADAPTER_TYPE3="HoulsbyConfig"
# ADAPTER_TYPE6="MAMConfig"


PARTITION=20
TRIPLE_PER_RELATION=5000
EPOCH=5
LR=1e-04

# python src/relation_prompt/run_pretrain.py \
# --model $MODEL \
# --tokenizer $TOKENIZER \
# --input_dir $INPUT_DIR \
# --output_dir $OUTPUT_DIR \
# --n_partition $PARTITION \
# --triple_per_relation $TRIPLE_PER_RELATION \
# --adapter_names  $ADAPTER_NAMES \
# --adapter_type $ADAPTER_TYPE \
# --use_adapter \
# --cuda \
# --num_workers 16 \
# --max_seq_length 64 \
# --batch_size 16 \
# --lr $LR \
# --epochs $EPOCH \
# --use_prompt \
# --random_partition \

python src/relation_prompt/run_pretrain.py \
--model $MODEL \
--tokenizer $TOKENIZER \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--n_partition $PARTITION \
--triple_per_relation $TRIPLE_PER_RELATION \
--adapter_names  $ADAPTER_NAMES \
--adapter_type $ADAPTER_TYPE \
--use_adapter \
--cuda \
--num_workers 16 \
--max_seq_length 64 \
--batch_size 16 \
--lr $LR \
--epochs $EPOCH \
# --use_prompt \

python src/relation_prompt/run_pretrain.py \
--model $MODEL \
--tokenizer $TOKENIZER \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--n_partition $PARTITION \
--triple_per_relation $TRIPLE_PER_RELATION \
--adapter_names  $ADAPTER_NAMES \
--adapter_type $ADAPTER_TYPE \
--use_adapter \
--cuda \
--num_workers 16 \
--max_seq_length 64 \
--batch_size 16 \
--lr $LR \
--epochs $EPOCH \
--use_prompt \