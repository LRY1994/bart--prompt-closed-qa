DATASET="WebQuestion"
# TriviaQA
# NaturalQuestions
# SQuAD2
MODEL_DIR="checkpoints/bart-base_20220515_233736_adapter"
DATA_DIR="/home/simon/datasets/WebQuestion/splitted/"
BASE_MODEL="facebook/bart-base"
T=1
LR=1e-5
TRAIN_MODE="fusion"
OUTPUT_DIR="output"

python src/evaluation/eval_webquestion.py \
--dataset $DATASET \
--train_mode $TRAIN_MODE \
--model_dir $MODEL_DIR \
--data_dir $DATA_DIR  \
--base_model $BASE_MODEL \
--tokenizer $BASE_MODEL  \
--adapter_num 50 \
--batch_size 8 \
--eval_batch_size 16 \
--max_output_length 5 \
--learning_rate $LR   \
--pretrain_epoch 0 \
--epochs 10 \
--repeat_runs 1 \
--temperature $T \
--output_dir  $OUTPUT_DIR  \
--cuda \

#bash src/evaluation/run_eval_wq.sh
#sudo wg-quick up tw
# git remote set-url origin https://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@github.com/LRY1994/knowledge-infusion.git/
