from email.mime import base
import os
import random
import shutil
import time
from argparse import ArgumentParser
from datetime import datetime
from os import listdir
from statistics import mean, stdev
from xmlrpc.client import boolean
import logging
import numpy as np

import torch
from transformers import (
    AdapterFusionConfig,
    AutoConfig, 
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
)

import wandb
from utils.bert_evaluator import BertEvaluator
from utils.bert_trainer import BertTrainer
from utils.bioasq_processor import BioAsqProcessor
from utils.common_utils import print_args_as_table

# wandb.init(project="webquestion")

def get_args():
    parser = ArgumentParser(
        description="Evaluate model on BioAsq 7b dataset from BlURB."
    )
    parser.add_argument(
        "--train_mode",
        default="base",
        type=str,
        required=True,
        help="three modes:  adapter, base",
    )
   
    parser.add_argument("--base_model", default=None, type=str, required=True)
    parser.add_argument("--dataset", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--tokenizer", default=None, type=str, required=False)
    parser.add_argument("--cuda", action="store_true", help="to use gpu")
    parser.add_argument("--amp", action="store_true", help="use auto mixed precision")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat_runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrain_epoch", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="t=1: softmax fusion, 0<t<1: gumbel softmax fusion, t<0: MOE",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1,
        help="training examples ratio to be kept.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--groups", type=str, default=None, help="groups to be chosen")

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--train_file", default="train.tsv")
    parser.add_argument("--dev_file", default="dev.tsv")
    parser.add_argument("--test_file", default="test.tsv")

    parser.add_argument(
        "--max_input_length",
        default=128,
        type=int,
        help="The maximum total input sequence length.",
    )
    parser.add_argument(
        "--max_output_length",
        default=128,
        type=int,
        help="The maximum total input sequence length.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )

    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--length_penalty', type=float, default=2.0)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_k', type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--use_multiprocessed_decoding', type=bool, default=False)

    args = parser.parse_args()
    return args


def evaluate_split(model, processor, tokenizer, args, logger,split="dev"):
    evaluator = BertEvaluator(model, processor, tokenizer, args, logger,split )
    result = evaluator.get_scores()
    split_result = {}
    for k, v in result.items():
        split_result[f"{split}_{k}"] = v
    return split_result


def get_tf_flag(args):
    from_tf = False
    # if (
    #     (
    #         ("BioRedditBERT" in args.model)
    #         or ("BioBERT" in args.model)
    #         or ("SapBERT" in args.model)
    #     )
    #     and "step_" not in args.model
    #     and "epoch_" not in args.model
    # ):
    #     from_tf = True

    # if ("SapBERT" in args.model) and ("original" in args.model):
    #     from_tf = False
    return from_tf


def search_adapters(args):
    """[Search the model_path, take all the sub directions as adapter_names]

    Args:
        args (ArgumentParser)

    Returns:
        [dict]: {model_path:[adapter_names]}
    """
    adapter_paths_dic = {}
   
    model_path = args.model_dir  # checkpoints/roberta-base_20220411_001827_adapter
    adapter_paths = [f for f in listdir(model_path)]#[group_0_epoch_0,xxx]
    print(f"Found {len(adapter_paths)} adapter paths")
    # model_path父目录, adapter_paths adpter.json
    adapter_paths = check_adapter_names(model_path, adapter_paths)
    adapter_paths_dic[model_path] = adapter_paths
    return adapter_paths_dic


def check_adapter_names(model_path, adapter_names):
    """[Check if the adapter path contrains the adapter model]

    Args:
        model_path ([type]): [description]
        adapter_names ([type]): [description]

    Raises:
        ValueError: [description]
    """
    checked_adapter_names = []
    print(f"Checking adapter namer:{model_path}:{len(adapter_names)}")
    for adapter_name in adapter_names:  
        adapter_model_path = os.path.join(model_path, adapter_name) # checkpoints/roberta-base_20220411_001827_adapter/group_0_epoch_0
        if f"epoch_{args.pretrain_epoch}" not in adapter_name:
            # check pretrain_epoch
            continue
        if args.groups and int(adapter_name.split("_")[1]) not in set(args.groups):
            # check selected groups
            continue
        adapter_model_path = os.path.join(adapter_model_path, "pytorch_adapter.bin") 
        assert os.path.exists(
            adapter_model_path
        ), f"{adapter_model_path} adapter not found."

        checked_adapter_names.append(adapter_name)
    print(f"Valid adapters ({len(checked_adapter_names)}):{checked_adapter_names}")
    return checked_adapter_names




def load_fusion_adapter_model(args,base_model):
    # print(base_model)
    """Load fusion adapter model.

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    adapter_names_dict = search_adapters(args)
    fusion_adapter_rename = []
    for model_path, adapter_names in adapter_names_dict.items():
        for adapter_name in adapter_names:
            adapter_dir = os.path.join(model_path, adapter_name)
            new_adapter_name = model_path[-14:][:-8] + "_" + adapter_name  
            print('before')        
            base_model.load_adapter(adapter_dir, load_as=new_adapter_name)###这里有问题
            print('after')
            fusion_adapter_rename.append(new_adapter_name)

    # print("fusion_adapter_rename:",fusion_adapter_rename)
    fusion_config = AdapterFusionConfig.load("dynamic", temperature=args.temperature)
    base_model.add_fusion(fusion_adapter_rename, fusion_config)
    base_model.set_active_adapters(fusion_adapter_rename)
    config = AutoConfig.from_pretrained(
        os.path.join(adapter_dir, "adapter_config.json")
    )
    # base_model.train_fusion([adapter_names])
    return config, base_model


if __name__ == "__main__":
    

    args = get_args()
    print(args)
    #### Start writing logs

    log_filename = "log.txt"
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Device: {} ".format(str(device).upper()))
    logger.info("Number of GPUs: {} ".format(n_gpu))
  

    train_acc_list = []
    dev_acc_list = []
    test_acc_list = []
    seed_list = []
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    
    # Record config on wandb
    # wandb.config.update(args)
    print_args_as_table(args)

    processor = BioAsqProcessor(args.data_dir, logger)   
    tokenizer = BartTokenizer.from_pretrained(args.base_model)

    for i in range(args.repeat_runs):
        logger.info( f'**Start the {i}th/{args.repeat_runs}(args.repeat_runs) training.****' )
    
        # Set random seed for reproducibility
        seed = int(time.time())
        logger.info(f"Generate random seed {seed}.")
        seed_list.append(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        args.best_model_dir = f"src/temp/model_{seed}/"
        os.makedirs(args.best_model_dir, exist_ok=True)
        basemodel = BartForConditionalGeneration.from_pretrained(args.base_model)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)
        if args.train_mode == "fusion":
            # args.base_model will be a folder of pre-trained models over partitions
            config, model = load_fusion_adapter_model(args, basemodel)
        elif args.train_mode == "base":
            # use base bart model
            config = BartConfig.from_pretrained(args.base_model) #AutoConfig.from_pretrained(args.base_model)
            model = basemodel

        

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
       

        logger.info('***Training Model***')
        trainer = BertTrainer(model, processor, tokenizer, args,logger)      
        trainer.train()
       

        # 只取最好的
        logger.info("***Evaluating Model(modal is set)***")
        logger.info(f"load model from {args.best_model_dir}model.bin")
        model = torch.load(args.best_model_dir + "model.bin")
     


        train_result = evaluate_split(model, processor, tokenizer, args, logger,split="train")
        train_result["run_num"] = i
        # wandb.log(train_result)  # Record Dev Result
        train_acc_list.append(train_result["train_correct_ratio"])

        dev_result = evaluate_split(model, processor, tokenizer, args, logger,split="dev")
        dev_result["run_num"] = i
        # wandb.log(dev_result)  # Record Dev Result
        dev_acc_list.append(dev_result["dev_correct_ratio"])

        test_result = evaluate_split(model, processor, tokenizer, args, logger,split="test")
        test_result["run_num"] = i
        # wandb.log(test_result)  # Record Testing Result
        test_acc_list.append(test_result["test_correct_ratio"])

        if (
            test_result["test_correct_ratio"] < 0.86
        ):  # keep the models with excellent performance
            shutil.rmtree(args.best_model_dir)#递归地删除文件
        else:
            logger.info(f"Saving model to {args.best_model_dir}.")
            logger.info(f"correct_ratio of {test_result['test_correct_ratio']}.")

    logger.info(f"***{args.repeat_runs} training is finished****")
    result = {}
    result["seed_list"] = seed_list
    result["train_acc_mean"] = mean(train_acc_list)  # average of the ten runs
    result["train_acc_std"] = stdev(train_acc_list)  # average of the ten runs
    result["dev_acc_mean"] = mean(dev_acc_list)  # average of the ten runs
    result["dev_acc_std"] = stdev(dev_acc_list)  # average of the ten runs
    result["test_acc_mean"] = mean(test_acc_list)  # average of the ten runs
    result["test_acc_std"] = stdev(test_acc_list)  # average of the ten runs
    # wandb.config.update(result)
    logger.info(result)
    
    print(result)