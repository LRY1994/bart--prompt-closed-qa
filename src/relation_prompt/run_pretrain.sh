MODEL="facebook/bart-base"
TOKENIZER="facebook/bart-base"
# MODEL="roberta-base"
# TOKENIZER="roberta-base"
INPUT_DIR="/home/simon/wikidata5m"
OUTPUT_DIR="checkpoints"
DATASET_NAME="wikidata5m"
ADAPTER_NAMES="entity_predict"
ADAPTER_TYPE="ParallelConfig" #PrefixTuningConfig HoulsbyConfig PfeifferConfig ParallelConfig
ADAPTER_TYPE2="PfeifferConfig"
ADAPTER_TYPE3="HoulsbyConfig"
PARTITION=20
TRIPLE_PER_RELATION=5000
EPOCH=5

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
--max_seq_length 128 \
--batch_size 64  \
--lr 1e-04 \
--epochs $EPOCH \

python src/relation_prompt/run_pretrain.py \
--model $MODEL \
--tokenizer $TOKENIZER \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--n_partition $PARTITION \
--triple_per_relation $TRIPLE_PER_RELATION \
--adapter_names  $ADAPTER_NAMES \
--adapter_type $ADAPTER_TYPE2 \
--use_adapter \
--cuda \
--num_workers 16 \
--max_seq_length 128 \
--batch_size 64  \
--lr 1e-04 \
--epochs $EPOCH \

python src/relation_prompt/run_pretrain.py \
--model $MODEL \
--tokenizer $TOKENIZER \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--n_partition $PARTITION \
--triple_per_relation $TRIPLE_PER_RELATION \
--adapter_names  $ADAPTER_NAMES \
--adapter_type $ADAPTER_TYPE3 \
--use_adapter \
--cuda \
--num_workers 16 \
--max_seq_length 128 \
--batch_size 64  \
--lr 1e-04 \
--epochs $EPOCH \
#bash src/relation_prompt/run_pretrain.sh
