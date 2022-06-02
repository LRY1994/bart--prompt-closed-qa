from unittest import result
import warnings
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


from tqdm import tqdm

from multiprocessing import Pool, cpu_count
# from .abstract_processor import convert_examples_to_features
from utils.data_processor import create_dataloader
import logging
import re
import string
from transformers.models.bart.modeling_bart import shift_tokens_right
# Suppress warnings from sklearn.metrics
warnings.filterwarnings("ignore")

class BertEvaluator(object):
    def __init__(
        self, model, processor, tokenizer, args, logger,split="dev",
        
    ):
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.logger = logger
        self.device = args.device
        self.model.eval()

       
        if (split == 'dev'):   self.eval_examples = self.processor.get_dev_examples()
        if (split == 'train'): self.eval_examples = self.processor.get_train_examples()
        if (split == 'test'):  self.eval_examples = self.processor.get_test_examples()

    def _get_inputs_dict(self, batch):
        device = self.device     
        pad_token_id = self.tokenizer.pad_token_id

        # source_ids, source_mask, y = batch[0], batch[1], batch[2]
        # y_ids = y[:, :-1].contiguous()
        # lm_labels = y[:, 1:].clone()
        # lm_labels[y[:, 1:] == pad_token_id] = -100       
        # inputs = {         
        #     "input_ids": source_ids.to(device),
        #     "attention_mask": source_mask.to(device),
        #     "decoder_input_ids": y_ids.to(device),
        #     "labels": lm_labels.to(device),
        # }
       
        decoder_start_token_id = self.model.config.decoder_start_token_id        
        input_ids, attention_mask, decoder_input_ids = batch[0], batch[1], batch[2]
        _decoder_input_ids = shift_tokens_right(decoder_input_ids, pad_token_id,decoder_start_token_id)#关建
        lm_labels = decoder_input_ids[:, :].clone()
        lm_labels[lm_labels[:, :] == pad_token_id] = -100

        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "decoder_input_ids": _decoder_input_ids.to(device),
            "labels": lm_labels.to(device),
        }
        return inputs

        
    def get_loss(self, silent=False,verbose=True,**kwargs):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        eval_dataloader = create_dataloader(
            self.eval_examples, 
            self.tokenizer, 
            self.args.batch_size, 
            self.args.max_input_length, 
            self.args.max_output_length, 
            isTraining=False)

        self.model.eval()

        eval_loss = 0
        nb_eval_steps = 0
       
        for batch in tqdm( eval_dataloader, desc="Evaluating", disable=silent): 
            inputs = self._get_inputs_dict( batch )       
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = outputs[0]
                eval_loss += loss.mean().item()
            nb_eval_steps += 1
    
        eval_loss = eval_loss / nb_eval_steps
       

        return eval_loss

    def get_accuracy(self,**kwargs):      
        result,preds = self.inference()   
        return result['correct_ratio']
       
    def get_scores(self, silent=False,verbose=True,**kwargs):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """
        eval_dataloader = create_dataloader(
            self.eval_examples, 
            self.tokenizer, 
            self.args.batch_size, 
            self.args.max_input_length, 
            self.args.max_output_length, 
            isTraining=False)

        self.model.eval()

        eval_loss = 0
        nb_eval_steps = 0
        results = {}
        
        for batch in tqdm( eval_dataloader, desc="Evaluating", disable=silent): 

            inputs = self._get_inputs_dict( batch ) 
         
            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs[0]
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            eval_loss += loss.item()
            nb_eval_steps += 1
            
        

                # outputs = self.model.model(**inputs)
                # lm_logits = F.linear(outputs[0], self.model.model.shared.weight, bias=self.model.final_logits_bias)
        
                # loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.model.config.pad_token_id)
                # loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size),decoder_input_ids.view(-1))
               

        # loss       
        eval_loss = eval_loss / nb_eval_steps
        results["eval_loss"] = eval_loss  
        # accuracy
        result , preds = self.inference()
        results.update(result)# {correct_num, correct_ratio}
        target_text = [d.target_text.replace('\n','') for d in self.eval_examples]
        result = self.compute_metrics(target_text, preds, **kwargs)
        results.update(result)# metrics

        # output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        # os.makedirs(self.args.output_dir, exist_ok=True)
        # with open(output_eval_file, "w") as writer:
        #     for key in sorted(results.keys()):
        #         writer.write("{} = {}\n".format(key, str(results[key])))

        

        return results #{eval_loss,predict_correct_num,predict_correct_ratio}
            

    def inference(self,  output_dir=None, suffix=None, verbose=True, silent=False):
        """
        Performs predictions on a list of text.
        Args:
            pred_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence.            
            output_dir: The directory where predictition results files will be saved. If not given, self.args.output_dir will be used.
            suffix: The supplementary suffix of prediction results name.
        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8" 
        self.model.eval()    
        pred_data = self.eval_examples 
        to_predict = [d.input_text.replace('\n','') for d in pred_data]
        target_predict = [d.target_text.replace('\n','') for d in pred_data]#groundtruth
        assert len(to_predict)==len(target_predict)
        # print('to_predict:',to_predict[0])
        # print('target_predict:',target_predict[0])

        
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)


        all_outputs = []
        # Batching
        for batch in tqdm(
            [to_predict[i : i + self.args.eval_batch_size] for i in range(0, len(to_predict), self.args.eval_batch_size)],
            desc='Predicting', 
            disable=silent, 
            mininterval=0,):
            
            question_input = self.tokenizer.batch_encode_plus(
                batch,
                max_length=self.args.max_output_length,
                padding=True,
                return_tensors="pt",
                truncation=True,
            )

            input_ids , attention_mask = question_input['input_ids'],question_input['attention_mask']
          
            outputs = self.model.generate(
                input_ids=input_ids.to(self.device) ,
                attention_mask=attention_mask.to(self.device) ,# 这个一定要加
                max_length=self.args.max_output_length,            
                early_stopping=self.args.early_stopping,
            )
           
            
            all_outputs.extend(outputs.cpu().numpy())

        if self.args.use_multiprocessed_decoding:
            self.model.to("cpu")
            with Pool(self.args.process_count) as p:
                outputs = list(
                    tqdm(
                        p.imap(self._decode, all_outputs, chunksize=self.args.multiprocessing_chunksize),
                        total=len(all_outputs),
                        desc="Decoding outputs",
                        disable=silent,
                    )
                )           
        else:
            outputs = [
                self.tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for output_id in all_outputs
            ]

        output_predication_file = os.path.join(output_dir, "{}_predictions.txt".format(self.args.dataset))
        correct_num = 0

        with open(output_predication_file, "w", encoding="utf8", errors="ignore") as writer:
            writer.write("to_predict\n\toutput\n\ttarget\n\tnomalize_output\n\tnomalize_target\n\t\EM\n")
            for i in range(len(outputs)):
                prediction = outputs[i].strip()       
                groudtruth = target_predict[i].strip().split('\t')
                flag = False   
                if get_exact_match(prediction, groudtruth):
                    print(prediction+'\n'+str(groudtruth))
                    correct_num += 1
                    flag = True

                writer.write(to_predict[i]+"\n\t"+prediction+"\n\t"+target_predict[i]+"\n\t"+str(flag) +"\n\n")
                
                    
                # writer.write(to_predict[i]+"\n\t"+outputs[i]+"\n\t"+target_predict[i]+"\n\t"+prediction+"\n\t"+str(groudtruth)+"\n\t"+str(flag) +"\n\n")


        correct_ratio = correct_num/float(len(outputs))
    
        if self.args.num_return_sequences > 1:
            outputs = [
                outputs[i : i + self.args.num_return_sequences]
                for i in range(0, len(outputs), self.args.num_return_sequences)
            ]
        else:
            outputs = outputs

        result = { 'correct_num':correct_num, 'correct_ratio':correct_ratio }
        return  result,outputs


    def compute_metrics(self, labels, preds, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            labels: List of target sequences
            preds: List of model generated outputs
            **kwargs: Custom metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.

        Returns:
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"
        # assert len(labels) == len(preds)

        results = {}

        for metric, func in kwargs.items():
            results[metric] = func(labels, preds)

        return results    
    

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_exact_match(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))