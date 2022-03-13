import datetime
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset
from tqdm import tqdm
import pandas as pd
from torch import Tensor, nn
# from .abstract_processor import convert_examples_to_features
from utils.bioasq_processor import get_inputs_dict,create_dataloader
from .bert_evaluator import BertEvaluator
from transformers.modeling_bart import shift_tokens_right
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

class BertTrainer(object):
    def __init__(self, model, processor, tokenizer, args):      
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = args.device
        self.train_examples = self.processor.get_train_examples()
        if args.train_ratio < 1:
            keep_num = int(len(self.train_examples) * args.train_ratio) + 1
            self.train_examples = self.train_examples[:keep_num]
            print(f"Reduce Training example number to {keep_num}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_str = self.args.model
        if "/" in model_str:
            model_str = model_str.split("/")[1]#bart-base

        self.num_train_optimization_steps = (
            int(
                len(self.train_examples)
                / args.batch_size
                / args.gradient_accumulation_steps
            )
            * args.epochs
        )

        self.log_header = (
            "Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss"
        )
        self.log_template = " ".join(
            "{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}".split(
                ","
            )
        )

        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_acc, self.unimproved_iters = -1, 0
        self.early_stop = False
        optimizer, scheduler = self.prepare_opt_sch( args)
        self.optimizer = optimizer
        self.scheduler = scheduler


    def prepare_opt_sch(self, args):
        """Prepare optimizer and scheduler.

        Args:
            model ([type]): [description]
            args ([type]): [description]

        Returns:
            [type]: [description]
        """
        train_examples = self.train_examples
        num_train_optimization_steps = (
            int(len(train_examples) / args.batch_size / args.gradient_accumulation_steps)
            * args.epochs
        )
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            weight_decay=0.01,
            correct_bias=False,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=num_train_optimization_steps,
            num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        )
        return optimizer, scheduler 

    def train_epoch(self, train_dataloader,epoch_number):
        self.tr_loss = 0
        train_losses = []

        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Running Epoch {epoch_number+1} of {self.args.epochs} ",
            mininterval=0,
        )
       
        
        for step, batch in enumerate(train_dataloader):
            self.model.train()
           
            inputs = get_inputs_dict(batch,self.device)         
            # decoder_input_ids = inputs['decoder_input_ids']
            # _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.model.config.pad_token_id) 
            inputs['decoder_input_ids'] = shift_tokens_right(inputs['decoder_input_ids'], self.model.config.pad_token_id)         
            outputs = self.model.model(**inputs)
          
            # print(outputs[0].shape)#torch.Size([4, 36, 1024]) 
            # print(self.model.shared.weight.shape)#torch.Size([50265, 1024])
            # print(self.final_logits_bias.shape)#torch.Size([1, 50265])

            lm_logits = F.linear(outputs[0], self.model.model.shared.weight, bias=self.model.final_logits_bias)
       
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.model.config.pad_token_id)
            loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size),
                              inputs['decoder_input_ids'].view(-1))
            
            
            # # model outputs are always tuple in pytorch-transformers (see doc)
            # loss = outputs[0]
            # print(type(loss))
            # # with open('ouput.txt','w') as f:
            # #     f.write(str(outputs))
            # # print('loss.item():',loss)
            # current_loss = loss.item()
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            batch_iterator.set_description(
                f"Epochs {epoch_number}/{self.args.epochs}. Running Loss: {loss:9.4f}"
            )


            
            train_losses.append(loss.detach().cpu())
            loss.backward()           
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1

        # print(train_losses)

    def train(self):
        # train_features = convert_examples_to_features(
        #     self.train_examples, self.args.max_input_length, self.args.max_output_length, self.tokenizer
        # )

          
        print("Number of train examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Num of steps:", self.num_train_optimization_steps)
       
        train_dataloader = create_dataloader(
            self.train_examples, 
            self.tokenizer, 
            self.args.batch_size, 
            self.args.max_input_length, 
            self.args.max_output_length, 
            isTraining=True)
    
      
        for epoch in tqdm(range(self.args.epochs), file=sys.stdout, desc="Epoch"):
            self.train_epoch(train_dataloader,epoch)

            dev_evaluator = BertEvaluator(
                self.model, self.processor, self.tokenizer, self.args, split="dev"
            )
            results = dev_evaluator.get_scores()
            
            

            # Update validation results
            if results["correct_ratio"] > self.best_dev_acc:
                self.unimproved_iters = 0
                self.best_dev_acc = results["correct_ratio"]
                torch.save(self.model, self.args.best_model_dir + "model.bin")
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    tqdm.write(
                        "Early Stopping. Epoch: {}, Best Dev performance: {}".format(
                            epoch, self.best_dev_acc
                        )
                    )
                    break
    
