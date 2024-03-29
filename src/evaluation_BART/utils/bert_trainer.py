from asyncio.log import logger
import datetime
import sys
import json
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset
from tqdm.auto import tqdm, trange
import pandas as pd
from torch import Tensor, nn
# from .abstract_processor import convert_examples_to_features
from utils.data_processor import create_dataloader
from .bert_evaluator import BertEvaluator
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

class BertTrainer(object):
    def __init__(self, model, processor, tokenizer, args,logger,wandb):      
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = args.device
        self.train_examples = self.processor.get_train_examples()
        self.logger = logger
        self.wandb = wandb
       
        # model_str = self.args.model
        # if "/" in model_str:
        #     model_str = model_str.split("/")[1]#bart-base

     
        self.log_header = (
            "Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss"
        )
        self.log_template = " ".join(
            "{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}".split(
                ","
            )
        )

        
        self.best_dev_acc, self.unimproved_iters = -1, 0
        self.early_stop = False
        optimizer, scheduler = self.prepare_opt_sch( args)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = 0
        self.epoch_losses = []


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
        self.num_train_optimization_steps = num_train_optimization_steps

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
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
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=num_train_optimization_steps,
            num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        )
        return optimizer, scheduler 
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


    def train_epoch(self, train_dataloader,epoch_number):
        tr_loss = 0
        
        logger.info(f"Running Epoch {epoch_number} of {self.args.epochs} ")
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Running Epoch {epoch_number} of {self.args.epochs} ",
            mininterval=0,
        )
       
        
        for step, batch in enumerate(batch_iterator):          
            inputs = self._get_inputs_dict( batch )   
            
            outputs = self.model(**inputs)
            loss = outputs[0]
            # loss.requires_grad_(True) 
           
            #outputs = self.model.model(**inputs)
            # print(outputs[0].shape)#torch.Size([4, 36, 768]) 
            # print(self.model.model.shared.weight.shape)#torch.Size([50265, 768])
            # print(self.model.final_logits_bias.shape)#torch.Size([1, 50265])

            # lm_logits = F.linear(outputs[0], self.model.model.shared.weight, bias=self.model.final_logits_bias)
       
            # loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.model.config.pad_token_id)
            # loss = loss_fct( lm_logits.view(-1, self.model.config.vocab_size), inputs['decoder_input_ids'].view(-1) )
            
           
            
            
            # current_loss = loss.item()
            
            # batch_iterator.set_description(
            #     f"Batch Running Loss: {current_loss:9.4f}"
            # )
            
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
                       
            loss.backward() 
            tr_loss += loss.item()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                self.global_step += 1
        
        logger.info(f"Epoch Training Loss :{tr_loss}")
       
        self.epoch_losses.append(tr_loss)
            
        return tr_loss



        # print(train_losses)

    def train(self):
         
        self.model.train()
        logger = self.logger
        args = self.args
            
        logger.info(f"Number of train examples: {len(self.train_examples)}")
        logger.info(f"Batch size:{args.batch_size}")
        logger.info(f"Num of steps:{self.num_train_optimization_steps}")
        
        
       
        train_dataloader = create_dataloader(
            self.train_examples, 
            self.tokenizer, 
            self.args.batch_size, 
            self.args.max_input_length, 
            self.args.max_output_length, 
            isTraining=True)
    
        train_iterator = tqdm(range(self.args.epochs), file=sys.stdout, desc="Epoch")
        
        for epoch in train_iterator:
            train_iterator.set_description(f"Epoch {epoch } of {args.epochs}")
            logger.info(f"Epoch {epoch } of {args.epochs}")
            tr_loss = self.train_epoch(train_dataloader,epoch)

            dev_evaluator = BertEvaluator(
                self.model, self.processor, self.tokenizer, self.args, logger,split="dev"
            )

           
            dev_loss = dev_evaluator.get_loss()            
            dev_accuracy = dev_evaluator.get_accuracy()                    
            logger.info(f"dev_results :{dev_loss}, {dev_accuracy}")   

            self.wandb.log(
                {'Epoch Dev accuracy':dev_accuracy,
                 'Epoch Dev Loss':dev_loss,
                 'Epoch Train Loss': tr_loss
                },
                step = epoch )
            
            

            # Update validation results
            if dev_accuracy > self.best_dev_acc:
                self.unimproved_iters = 0
                self.best_dev_acc = dev_accuracy

                test_evaluator = BertEvaluator(
                    self.model, self.processor, self.tokenizer, self.args, logger,split="test"
                )
                test_accuracy= test_evaluator.get_accuracy()
                logger.info(f"test_accuracy :{test_accuracy}") 
                self.wandb.log(
                    {'Epoch Test accuracy':test_accuracy},
                    step = epoch )

                logger.info(f"save model to {self.args.best_model_dir}model.bin")         
                # torch.save(self.model, self.args.best_model_dir + "model.bin")
                self.model.save_pretrained(self.args.best_model_dir)
                if self.args.train_mode == "fusion": 
                    self.model.save_adapter_fusion(self.args.best_model_dir, self.args.fusion_adapter_rename)
                    self.model.save_all_adapters(self.args.best_model_dir)

            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    tqdm.write(
                        "Early Stopping. Epoch: {}, Best Dev performance: {}".format(
                            epoch, self.best_dev_acc
                        )
                    )
                    logger.info("Early Stopping. Epoch: {}, Best Dev performance: {}".format(
                            epoch, self.best_dev_acc
                        ))
                    break
                    
            
            logger.info(f"Epoch ALL Loss :{self.epoch_losses}")        