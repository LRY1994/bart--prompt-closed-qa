DATASET="TriviaQA"
DATA_DIR="/home/simon/datasets/TriviaQA/splitted/"

MODEL_DIR="checkpoints/bart-base_ParallelConfig_adapter"
MODEL_DIR2="checkpoints/bart-base_PfeifferConfig_adapter"
MODEL_DIR3="checkpoints/bart-base_HoulsbyConfig_adapter"
BASE_MODEL="facebook/bart-base"
T=1
LR=4e-5
TRAIN_MODE="fusion"
TRAIN_MODE2="base"
OUTPUT_DIR="output"
TRAIN_BATCH_SIZE=16
PRE_EPOCH=4
EPOCH=20


#base
python src/evaluation_BART/eval_question.py \
--dataset $DATASET \
--train_mode  $TRAIN_MODE2 \
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
--pretrain_epoch $PRE_EPOCH \
--epochs $EPOCH \
--repeat_runs 1 \
--temperature $T \
--output_dir $OUTPUT_DIR \
--gradient_accumulation_steps 4 \
--cuda \

#fusion
# python src/evaluation_BART/eval_question.py \
# --dataset $DATASET \
# --train_mode $TRAIN_MODE \
# --model_dir $MODEL_DIR \
# --data_dir $DATA_DIR  \
# --base_model $BASE_MODEL \
# --tokenizer $BASE_MODEL  \
# --adapter_num 20 \
# --batch_size $TRAIN_BATCH_SIZE \
# --eval_batch_size $TRAIN_BATCH_SIZE \
# --max_input_length 64 \
# --max_output_length 64 \
# --learning_rate $LR   \
# --pretrain_epoch $PRE_EPOCH \
# --epochs $EPOCH \
# --repeat_runs 1 \
# --temperature $T \
# --output_dir $OUTPUT_DIR \
# --gradient_accumulation_steps 4 \
# --cuda \

# python src/evaluation_BART/eval_question.py \
# --dataset $DATASET \
# --train_mode $TRAIN_MODE \
# --model_dir $MODEL_DIR2 \
# --data_dir $DATA_DIR  \
# --base_model $BASE_MODEL \
# --tokenizer $BASE_MODEL  \
# --adapter_num 20 \
# --batch_size $TRAIN_BATCH_SIZE \
# --eval_batch_size $TRAIN_BATCH_SIZE \
# --max_input_length 64 \
# --max_output_length 64 \
# --learning_rate $LR   \
# --pretrain_epoch $PRE_EPOCH \
# --epochs $EPOCH \
# --repeat_runs 1 \
# --temperature $T \
# --output_dir $OUTPUT_DIR \
# --gradient_accumulation_steps 4 \
# --cuda \

# python src/evaluation_BART/eval_question.py \
# --dataset $DATASET \
# --train_mode $TRAIN_MODE \
# --model_dir $MODEL_DIR3 \
# --data_dir $DATA_DIR  \
# --base_model $BASE_MODEL \
# --tokenizer $BASE_MODEL  \
# --adapter_num 20 \
# --batch_size $TRAIN_BATCH_SIZE \
# --eval_batch_size $TRAIN_BATCH_SIZE \
# --max_input_length 64 \
# --max_output_length 64 \
# --learning_rate $LR   \
# --pretrain_epoch $PRE_EPOCH \
# --epochs $EPOCH \
# --repeat_runs 1 \
# --temperature $T \
# --output_dir $OUTPUT_DIR \
# --gradient_accumulation_steps 4 \
# --cuda \

#bash src/evaluation_BART/run_eval_wq.sh
#sudo wg-quick up tw
# git remote set-url origin https://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@github.com/LRY1994/knowledge-infusion.git/

