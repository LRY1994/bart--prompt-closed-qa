import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

from utils.abstract_processor import (
    BertProcessor,
    InputExample,
)
from utils.common import _construct_adj, partition_graph, timeit



class KGProcessor_prompt(BertProcessor):
    def __init__(
        self,
        data_dir,
        sub_set=1,
        name="Node Prediction With Partition",
        n_partition=50,
        triple_per_relation=5000,
        bi_direction=True,
        sub_group_idx=None,
        shuffle_rate=None,
    ):
        self.NAME = name
        self.id2ent = {}
        self.id2rel = {}
        self.n_partition = n_partition
        self.triple_per_relation = triple_per_relation
        self.sub_set = sub_set
        
       
        self.tri_file = os.path.join(data_dir, "wikidata5m_transductive_train.txt")
        self.ent_file = os.path.join(data_dir, "wikidata5m_entity.txt")
        self.rel_file = os.path.join(data_dir, "wikidata5m_relation.txt")
        if shuffle_rate:
            self.partition_file = os.path.join(
                data_dir, f"partition_{n_partition}_shuf_{shuffle_rate}.txt"
            )
            print(self.partition_file)
            assert os.path.exists(self.partition_file)
        else:
            self.partition_file = os.path.join(data_dir, f"partition_{n_partition}.txt")
        if sub_group_idx is not None:
            self.partition_file = os.path.join(
                data_dir, f"partition_{n_partition}_{sub_group_idx}_{n_partition}.txt"
            )
        self.bi_direction = bi_direction
        self.cache_feature_base_dir = os.path.join(
            data_dir, "feature_cache_metis_partition/"
        )
        os.makedirs(self.cache_feature_base_dir, exist_ok=True)
        self.examples_cache = {}  # Memory cache of loaded partion nodes
        self.load_data(sub_set)
        ## Wether to prediction both the head and tail nodes
        ## if bi_direction is False, only predict the tail node
        super(KGProcessor_prompt, self).__init__()


    def sample_triple(self, top_n):
        import heapq
        n_top_rel = list(heapq.nlargest(top_n, list(self.triple_list.items()), key=lambda s: len(s[1])))       
        top_rel = [id for id, tlist in n_top_rel]
        tri_per_rel =  [len(tlist) for id, tlist in n_top_rel][-1]#the least
        return top_rel , tri_per_rel


    def load_data(self, sub_set):

        ## Read entity file
        with open(self.ent_file, "r") as f:
            self.ent_total = 0
            if sub_set != 1:
                self.ent_total = int(self.ent_total * sub_set)

            for ent in f.readlines():
                self.ent_total += 1
                self.id2ent[ent.split("\t")[0].strip()] = ent.split("\t")[1]#'</s>'.join(ent.split("\t")[1:]).strip()#ent.split("\t")[1]
            print(
                f"Loading entities (subset mode:{sub_set}) ent_total:{self.ent_total} len(self.id2ent): {len(self.id2ent)}"
            )

        ## Read Relation File
        with open(self.rel_file, "r") as f:
            # print("Read Relation File")
            # self.rel_total = (int)(f.readline())  # num of total relations
            for rel in f.readlines():
                # arr = rel.split("\t")[1:]
                # result = arr[0].strip()
                # for index in range(1,len(arr)):
                #     tmp = result + ' </s> '+ arr[index].strip()
                #     if len(tmp.split(' ')) < 65: result = tmp
                #     else : break
                # self.id2rel[rel.split("\t")[0].strip()] = result
                self.id2rel[rel.split("\t")[0].strip()] = rel.split("\t")[1]
            print(f"{len(self.id2rel)} relations loaded.")
        
        # print(self.id2rel['P1001'])
        # print(self.id2ent['Q47551'])

          

        ## Read triple File
        f = open(self.tri_file, "r")
        # triples_total = (int)(f.readline())

        count = 0
        # self.triple_list = [[] for i in range(len(self.id2rel))]
        self.triple_list = {}
        for line in f.readlines():
            h, r, t = line.strip().split("\t")
            if (
                (h in self.id2ent)
                and (t in self.id2ent)
                and (r in self.id2rel)
            ):
                if r in self.triple_list:
                    self.triple_list[r].append((h, t))
                else:
                    self.triple_list[r] = [(h, t)]
                count += 1
        f.close()


        # top relation
        self.top_rel , tri_per_rel = self.sample_triple(self.n_partition)
        # print([self.id2rel[r] for r in self.top_rel])
        self.triple_list = {r: self.triple_list[r] for r in self.top_rel}
        
        import random
        self.triple_list = {k: random.sample(v, tri_per_rel)  if len(v) > tri_per_rel else v  for (k, v) in self.triple_list.items()}


    @timeit
    def _create_examples(self, group_idx):
        # group_idx = self.top_rel[group_idx]
        if group_idx in self.examples_cache:
            print(
                f"Get cache examples from partition {group_idx}/{self.n_partition} set"
            )
            return self.examples_cache[group_idx]
        examples = []

        
        for (h_id, t_id) in self.triple_list[self.top_rel[group_idx]]:
            text_h = self.id2ent[h_id]
            text_t = self.id2ent[t_id]
            text_r = self.id2rel[self.top_rel[group_idx]]
           
            examples.append(  # use text_h+test_r to predict t_id
                InputExample(
                    guid=None,
                    text_e=text_h + '<mask>',
                    text_r=text_r,
                    label=text_t,
                )
            )
        self.examples_cache[group_idx] = examples
        print(
            f"Get {len(examples)} examples of {self.NAME} datasets from partition '{text_r}' {group_idx}/{self.n_partition} set"
        )
        return examples