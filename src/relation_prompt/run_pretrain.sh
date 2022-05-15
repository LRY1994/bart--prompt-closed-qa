MODEL="facebook/bart-base"
TOKENIZER="facebook/bart-base"
# MODEL="roberta-base"
# TOKENIZER="roberta-base"
INPUT_DIR="/home/simon/wikidata5m"
OUTPUT_DIR="checkpoints"
DATASET_NAME="wikidata5m"
ADAPTER_NAMES="entity_predict"
PARTITION=100
TRIPLE_PER_RELATION=5000

python src/relation_prompt/run_pretrain.py \
--model $MODEL \
--tokenizer $TOKENIZER \
--input_dir $INPUT_DIR \
--output_dir $OUTPUT_DIR \
--n_partition $PARTITION \
--triple_per_relation $TRIPLE_PER_RELATION \
--adapter_names  $ADAPTER_NAMES \
--use_adapter \
--cuda \
--num_workers 16 \
--max_seq_length 128 \
--batch_size 64  \
--lr 1e-04 \
--epochs 1 \
--save_step 2000 


#bash src/relation_prompt/run_pretrain.sh
