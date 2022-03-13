import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from .abstract_processor import BertProcessor, InputExample

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

def load_bioasq(data_dir):
    train_df = read_data_source_target(data_dir + "train.source", data_dir + "train.target")   
    dev_df = read_data_source_target(data_dir + "dev.source", data_dir + "dev.target")
    test_df =  read_data_source_target(data_dir + "test.source", data_dir+ "test.target")
    return train_df, dev_df, test_df

def convert_examples_to_features(
    examples, max_input_length,max_output_length ,tokenizer, do_lowercase=False,append_another_bos=False
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

    questions = [d.input_text.replace('\\n','')  for d in examples]  
    answers = [d.target_text.replace('\\n','') for d in examples]

    # print(len(answers))
    # answers, metadata = self.flatten(answers)
    if do_lowercase:
        questions = [question.lower() for question in questions]
        answers = [answer.lower() for answer in answers]
    if append_another_bos:
        questions = ["<s> "+question for question in questions]
        answers = ["<s> " +answer for answer in answers]
    question_input = tokenizer.batch_encode_plus(questions,
                                                padding='max_length',
                                                max_length=max_input_length,
                                                truncation=True
                                                )
    answer_input = tokenizer.batch_encode_plus(answers,                                           
                                                padding=True,  
                                                max_length=max_output_length,                                          
                                                truncation=True
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

def get_inputs_dict(batch,device):     
    inputs = {
        'input_ids':batch[0].to(device),#torch.Size([4, 512])
        'attention_mask':batch[1].to(device),#torch.Size([4, 512])
        'encoder_outputs':None,
        'decoder_input_ids':batch[2].to(device),#torch.Size([4, 512])
        'decoder_attention_mask':batch[3].to(device),#torch.Size([4, 512])
        'decoder_cached_states':None,
        'use_cache':False            
    }
    return inputs

def create_dataloader(examples, tokenizer, batch_size, max_input_length, max_output_length, isTraining=False):
    features = convert_examples_to_features(
        examples, 
        max_input_length, 
        max_output_length, 
        tokenizer,
        do_lowercase=True,
        append_another_bos=True
    )       
    padded_input_ids = torch.LongTensor(features['input_ids'])
    padded_attention_mask = torch.LongTensor(features['attention_mask'])
    padded_decoder_input_ids= torch.LongTensor(features['decoder_input_ids'])
    padded_decoder_attention_mask= torch.LongTensor(features['decoder_attention_mask'])

    dataset = TensorDataset(
        padded_input_ids, 
        padded_attention_mask, 
        padded_decoder_input_ids,
        padded_decoder_attention_mask
    )

    if isTraining: 
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=batch_size
    )
    return dataloader

class BioAsqProcessor(BertProcessor):
    NAME = "webquestion"
    
    def __init__(self, data_dir):
        self.train_df, self.dev_df, self.test_df = load_bioasq(data_dir)

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
        print(
            f"Get {len(examples)} examples of {self.NAME} datasets for {set_type} set"
        )
        return examples



