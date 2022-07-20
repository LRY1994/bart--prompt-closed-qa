DATASET="NaturalQuestion"
DATA_DIR="/home/simon/datasets/NaturalQuestion/"
MODEL_DIR_TRUE="checkpoints/bart-base_PfeifferConfig_adapter_use_prompt_True"
MODEL_DIR_FALSE="checkpoints/bart-base_PfeifferConfig_adapter_use_prompt_False"
BASE_MODEL="facebook/bart-base"
OUTPUT_DIR="output"
TRAIN_MODE_FUSION="fusion"
TRAIN_MODE_BASE='base'
LR=1e-5
PARTITION=20
TRAIN_BATCH_SIZE=16
PRE_EPOCH=0
EPOCH=20
T=1
#base
python src/evaluation_BART/eval_question.py \
--dataset $DATASET \
--train_mode  $TRAIN_MODE_BASE\
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--adapter_num $PARTITION \
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


# FUSION NO PROMPT pretrain_epoch 
python src/evaluation_BART/eval_question.py \
--dataset $DATASET \
--train_mode $TRAIN_MODE_FUSION \
--model_dir $MODEL_DIR_FALSE \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--adapter_num $PARTITION \
--batch_size $TRAIN_BATCH_SIZE \
--eval_batch_size $TRAIN_BATCH_SIZE \
--max_input_length 64 \
--max_output_length 64 \
--learning_rate $LR   \
--pretrain_epoch 0 \
--epochs $EPOCH \
--repeat_runs 1 \
--temperature $T \
--output_dir $OUTPUT_DIR \
--gradient_accumulation_steps 4 \
--cuda \


# FUSION PROMPT pretrain_epoch 
python src/evaluation_BART/eval_question.py \
--dataset $DATASET \
--train_mode $TRAIN_MODE_FUSION \
--model_dir $MODEL_DIR_TRUE \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--adapter_num $PARTITION \
--batch_size $TRAIN_BATCH_SIZE \
--eval_batch_size $TRAIN_BATCH_SIZE \
--max_input_length 64 \
--max_output_length 64 \
--learning_rate $LR   \
--pretrain_epoch 0 \
--epochs $EPOCH \
--repeat_runs 1 \
--temperature $T \
--output_dir $OUTPUT_DIR \
--gradient_accumulation_steps 4 \
--cuda \
