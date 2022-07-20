
import pandas as pd

train_target= '/home/simon/datasets/TriviaQA/original/train.target'
file_target = open(train_target, 'r', encoding='utf8')
target = file_target.readlines()

train_source= '/home/simon/datasets/TriviaQA/original/train.source'
file_source = open(train_source, 'r', encoding='utf8')
source = file_source.readlines()

with open(train_source, "w", encoding="utf8", errors="ignore") as writer_source:
    with open(file_source, "w", encoding="utf8", errors="ignore") as writer_target:
        for i in range(0, len(target)):
            answers = target[i].split('\t')            
            for an in answers:
                writer_source.write(source[i])
                writer_target.write(an.replace('\n','')+'\n')



   