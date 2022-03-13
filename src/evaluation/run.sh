DATASET="WebQuestion"
# TriviaQA
# NaturalQuestions
# SQuAD2
MODEL_DIR="src/train_adpter/relation_prompt/checkpoints/"
DATA_DIR="/home/simon/datasets/WebQuestion/splitted/"
BASE_MODEL="facebook/bart-base"
MODEL="WebQuestion"
T=1
LR=1e-5
TRAIN_MODE="base"
OUTPUT_DIR="output"
python src/evaluation/eval_webquestion.py \
--train_mode $TRAIN_MODE \
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--model $MODEL  \
--batch_size 16 \
--eval_batch_size 16 \
--lr $LR   \
--pretrain_epoch 0 \
--epochs 10 \
--repeat_runs 5 \
--temperature $T \
--output_dir  $OUTPUT_DIR  \
--cuda \

#bash src/evaluation/run.sh
#sudo wg-quick up tw
# git remote set-url origin https://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@github.com/LRY1994/knowledge-infusion.git/