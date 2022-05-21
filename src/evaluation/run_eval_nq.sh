DATASET="NaturalQuestion"
# MODEL_DIR="checkpoints/bart-base_20220516_013128_ParallelConfig_adapter"
MODEL_DIR="checkpoints/bart-base_20220521_122004_PfeifferConfig_adapter"
DATA_DIR="/home/simon/datasets/NaturalQuestion/"
BASE_MODEL="facebook/bart-base"
T=1
LR=4e-5
TRAIN_MODE="fusion"
OUTPUT_DIR="output"
TRAIN_BATCH_SIZE=32

python src/evaluation/eval_question.py \
--dataset $DATASET \
--train_mode $TRAIN_MODE \
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--adapter_num 20 \
--batch_size $TRAIN_BATCH_SIZE \
--eval_batch_size $TRAIN_BATCH_SIZE \
--max_input_length 64 \
--max_output_length 64 \
--learning_rate $LR   \
--pretrain_epoch 5 \
--epochs 30 \
--repeat_runs 1 \
--temperature $T \
--output_dir $OUTPUT_DIR \
--gradient_accumulation_steps 4 \
--cuda \

#bash src/evaluation/run_eval_wq.sh
#sudo wg-quick up tw
# git remote set-url origin https://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@github.com/LRY1994/knowledge-infusion.git/

