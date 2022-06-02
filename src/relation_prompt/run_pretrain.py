from ast import If
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from transformers import (
    AdamW,
    AdapterConfig,
    HoulsbyConfig,
    PfeifferConfig,
    ParallelConfig,
    PrefixTuningConfig,
    AdapterType,
    AutoConfig, 
    AutoTokenizer,
    BartTokenizer,
    BartConfig ,T5Config,
    AutoModelForCausalLM
)
# from transformers.adapters import PrefixTuningConfig
import wandb

from utils.bert_trainer_prompt import BertTrainer
from utils.common import print_args_as_table
from utils.kg_processor import  KGProcessor_prompt
from model_BART import RelPromptBart
from model_T5 import RelPromptT5





from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Relation and Entity prediction.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="use GPU?",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--model", default="roberta-base", type=str)
    parser.add_argument("--tokenizer", default=None, type=str, required=False)
    parser.add_argument("--subset", default=1.0, type=float, required=False)
    parser.add_argument("--use_adapter", action="store_true", help="use adapters?")
    parser.add_argument("--shuffle_rate", type=str, default=None)
    parser.add_argument(
        "--cache_token_encodings",
        action="store_true",
        help="use cached tokenized encodings?",
    )
    parser.add_argument(
        "--non_sequential",
        action="store_true",
        help="if true, will initial a new model for each group",
    )
    parser.add_argument("--adapter_names", default=None, type=str, required=False)
    parser.add_argument(
        "--input_dir",
        default="/home/gzcheng/Projects/mop/kg_dir/wikidata5m_alias",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="models/kg_bert/",
        type=str,
        required=False,
    )
    parser.add_argument("--trained_model", default=None, type=str)
    parser.add_argument("--amp", action="store_true", help="use auto mixed precision")
    parser.add_argument("--save_step", default=2000, type=int, required=False)
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for",
    )
    parser.add_argument(
        "--CRate",
        default=8,
        type=int,
        help="adapter_reduction_factor, #{2,16,64}",
    )
    parser.add_argument(
        "--n_partition",
        default=50,
        type=int,
        help="Number of groups when partitioning graph",
    )
    parser.add_argument(
        "--sub_group_idx",
        default=None,
        type=int,
        help="Index of sub-groups of certain partitions",
    )
    parser.add_argument(
        "--bi_direction",
        action="store_true",
        help="Do bi-direction prediction for both head and tail nodes?",
    )
    parser.add_argument(
        "--adapter_layers",
        default=None,
        type=str,
        help="layers string for deploying adapters,e.g. 1,2,3, if None, will deploy adapters in all the layers",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--triple_per_relation",
        type=int,
        default=1000,
        help="Number of triple of one relation",
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        default='ParallelConfig',
        help="adapter_type",
    )


  

    args = parser.parse_args()
    return args



def init_model(args, relid=None):
    print(f"Initializing model from {args.model}")

    # if args.model.index('bart') :
    config = BartConfig.from_pretrained(args.model)   
    model = RelPromptBart.from_pretrained(  args.model , config=config, rel=relid, devices=args.device )  
    # if args.model.index('t5') :
    #     config = T5Config.from_pretrained(args.model)   
    #     model = RelPromptT5.from_pretrained(  args.model , config=config, rel=relid, devices=args.device )
  
    if args.use_adapter:
        # PfeifferConfig :places an adapter layer only after the feed-forward block in each Transformer layer.
        # adapter_config = PfeifferConfig(           
        #     # non_linearity=adapter_args.adapter_non_linearity,
        #     reduction_factor=args.CRate,  # adapter_args.adapter_reduction_factor, #{2,16,64}
        #     leave_out=[]           
        # )
        if args.adapter_type =='PrefixTuningConfig' : adapter_config= PrefixTuningConfig(flat=False, prefix_length=30)
        if args.adapter_type =='HoulsbyConfig' : adapter_config= HoulsbyConfig()
        if args.adapter_type =='PfeifferConfig' : adapter_config= PfeifferConfig()
        if args.adapter_type =='ParallelConfig' : adapter_config= ParallelConfig()

        model.add_adapter( args.adapter_names, config=adapter_config )
        model.train_adapter( args.adapter_names)
    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
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
    return model, optimizer

if __name__ == "__main__":
    # Set default configuration in args.py
    args = get_args()
    # 1. Start a W&B run
    wandb.init(project=f"Entity prediction with partition-{args.adapter_type}")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")  
    n_gpu = torch.cuda.device_count() if args.cuda else 0


    model_str = args.model
    if "/" in model_str:
        model_str = model_str.split("/")[1]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.model_str = f"{model_str}_{args.adapter_type}"
    if args.use_adapter:
        args.model_str += "_adapter"
    args.save_path = args.output_dir + '/' + args.model_str
    os.makedirs(args.save_path, exist_ok=True)
    

    print("Device:", str(device).upper())
    print("Number of GPUs:", n_gpu)
    print_args_as_table(args)
    # Set random seed for reproducibility
    if args.seed is None:
        args.seed = int(time.time())
        print(f"generate random seed {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    data_processor = KGProcessor_prompt(
        args.input_dir,
        args.subset,
        n_partition=args.n_partition,
        triple_per_relation=args.triple_per_relation,
        bi_direction=args.bi_direction,
        sub_group_idx=args.sub_group_idx,
        shuffle_rate=args.shuffle_rate,
    )
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.is_multilabel = False
    args.is_hierarchical = False
    args.adapter_layers = (
        None
        if args.adapter_layers == None
        else [int(i) for i in args.adapter_layers.split(",")]
    )
    if args.tokenizer is None:
        args.tokenizer = args.model
    wandb.config.update(args)

    
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer) 
  
    
    rel_names = list(map(data_processor.id2rel.get, data_processor.top_rel))
    print(rel_names)#['instance of', 'languages spoken, written or signed', 'director', 'country of citizenship', 'member of sports team', 'located in the administrative territorial entity', 'place of birth', 'followed by', 'cast member', 'exhibition history']
    # relations = tokenizer(rel_names, add_special_tokens=False)['input_ids']
    relations = tokenizer(rel_names, add_special_tokens=False, add_prefix_space=True)['input_ids']
   
    print(relations[0]) #[48768, 9]
    
    # model, optimizer = init_model(args, relations[0])
    for group_idx in range(args.n_partition):
        # if group_idx != 0 and args.non_sequential:
        #     model, optimizer = init_model(args, relations[group_idx])
        # if group_idx != 0 and args.non_sequential:
        model, optimizer = init_model(args, relations[group_idx])
        wandb.watch(model)
        trainer = BertTrainer(model, optimizer, data_processor, tokenizer, args)
        
        if args.cache_token_encodings:
            trainer.train_subgraph_cache_tokens(group_idx)
        else:
            trainer.train_subgraph(group_idx)
