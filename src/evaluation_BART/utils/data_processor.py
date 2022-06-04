import pandas as pd
import json
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, input_text, target_text=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        
        self.input_text = input_text
        self.target_text = target_text

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, decoder_input_ids,decoder_attention_mask):
        self.input_ids=input_ids
        self.attention_mask=attention_mask
        self.decoder_input_ids=decoder_input_ids
        self.decoder_attention_mask=decoder_attention_mask
        

def read_data_source_target(file_name_source, file_name_target):
    file_source = open(file_name_source, 'r', encoding='utf8')
    file_target = open(file_name_target, 'r', encoding='utf8')

    source = file_source.readlines()
    target = file_target.readlines()

    if len(source) != len(target):
        raise ValueError(
            "The length of the source file should be equal to target file"
        )
    length = len(source)
    source_target_pair = [[ source[i], target[i]] for i in range(length)] # "" for "prefix" used in t5_util.py
    data_df = pd.DataFrame(source_target_pair, columns=[ "input_text", "target_text"])
    return data_df

def load_data(data_dir):
    train_df = read_data_source_target(data_dir + "train.source", data_dir + "train.target")   
    dev_df = read_data_source_target(data_dir + "dev.source", data_dir + "dev.target")
    test_df =  read_data_source_target(data_dir + "test.source", data_dir+ "test.target")
    return train_df, dev_df, test_df

def convert_examples_to_features(
    examples, max_input_length,max_output_length ,tokenizer, do_lowercase=True,append_another_bos=True
):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """
    print ("Start tokenizing...")

    questions = [d.input_text.replace('\n','')  for d in examples] 
    answers = [d.target_text.replace('\n','') for d in examples]

    if do_lowercase:
        questions = [question.lower() for question in questions]
        answers = [answer.lower() for answer in answers]
    if append_another_bos:
        questions = ["<s> "+question for question in questions]
        answers = ["<s> " +answer for answer in answers]
  
    question_input = tokenizer.batch_encode_plus(questions,
                                                pad_to_max_length=True, 
                                                max_length=max_input_length ,
                                                truncation=True,
                                                return_tensors="pt"
                                                )
    answer_input = tokenizer.batch_encode_plus(answers,                                           
                                                pad_to_max_length=True, 
                                                max_length=max_output_length, 
                                                truncation=True,
                                                return_tensors="pt"
                                                )

    input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
    decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]


    # preprocessed_data = [input_ids, attention_mask,
    #                                  decoder_input_ids, decoder_attention_mask,
    #                                  ]
    # with open('train-barttokenized.json', "w") as f:
    #     json.dump([input_ids, attention_mask,
    #                 decoder_input_ids, decoder_attention_mask
    #                 ], f)


    return{
        'input_ids':input_ids,
        'attention_mask':attention_mask,
        'decoder_input_ids':decoder_input_ids,
        'decoder_attention_mask':decoder_attention_mask
    }



def create_dataloader(examples, tokenizer, batch_size, max_input_length, max_output_length, isTraining=False):
    features = convert_examples_to_features(
        examples, 
        max_input_length, 
        max_output_length, 
        tokenizer,
        do_lowercase=False,
        append_another_bos=False
    )       
    
    dataset = TensorDataset(
        torch.LongTensor(features['input_ids']) ,
        torch.LongTensor(features['attention_mask']),
        torch.LongTensor(features['decoder_input_ids']),
        torch.LongTensor(features['decoder_attention_mask'])
    )

    if isTraining: 
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=batch_size,
        num_workers=0
    )
    return dataloader

class DataProcessor():
    def __init__(self, data_dir,logger):
        self.train_df, self.dev_df, self.test_df = load_data(data_dir)
        self.logger = logger

    def get_train_examples(self):
        return self._create_examples(self.train_df, set_type="train")

    def get_dev_examples(self):
        return self._create_examples(self.dev_df, set_type="dev")

    def get_test_examples(self):
        return self._create_examples(self.test_df, set_type="test")

    def _create_examples(self, data_df, set_type):
        examples = []
        
        for (i, row) in data_df.iterrows():           
            input_text = row["input_text"]
            target_text = row["target_text"]
           
            examples.append(
                InputExample(input_text=input_text, target_text=target_text)
            )
        if self.logger:
             self.logger.info(
                f"Get {len(examples)} examples  for {set_type} set"
            )
        return examples



