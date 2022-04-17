DATASET="WebQuestion"
# TriviaQA
# NaturalQuestions
# SQuAD2
MODEL_DIR="src/train_adapter/relation_prompt/checkpoints/roberta-base_20220411_001827_adapter"
DATA_DIR="/home/simon/datasets/WebQuestion/splitted/"
BASE_MODEL="facebook/bart-base"
T=1
LR=1e-5
TRAIN_MODE="base"
OUTPUT_DIR="output"

python src/evaluation/eval_webquestion.py \
--dataset $DATASET \
--train_mode $TRAIN_MODE \
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--batch_size 8 \
--eval_batch_size 16 \
--learning_rate $LR   \
--pretrain_epoch 0 \
--epochs 10 \
--repeat_runs 2 \
--temperature $T \
--output_dir  $OUTPUT_DIR  \
--cuda \

#bash src/evaluation/run.sh
#sudo wg-quick up tw
# git remote set-url origin https://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@github.com/LRY1994/knowledge-infusion.git/

