from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration,BartConfig,BartTokenizer,BartAdapterModel

class RelPromptBart(BartAdapterModel):
    
    def __init__(self,config, rel=None, devices=None, use_prompt=True,prompt_length=64):
       
        super().__init__(config)
        self.prompt_length = len(rel)#
        print('prompt_length:')
        print(prompt_length)
        self.devices = devices
        self.word_embeddings  = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        if use_prompt :
            self.prompt = nn.Parameter( self.word_embeddings(torch.tensor(rel)).clone() )# initilize with token
            self.prompt.requires_grad = True
        else :
            self.prompt = torch.tensor(self.word_embeddings(torch.tensor(rel)).clone() )
        
        self.apply(self._init_weights)

        
       

    def forward(
            self,
            input_ids=None,
            mode='query'
    ):
        
        # word_embeddings 会自动加上一个终止符，
        ent_embedding = self.word_embeddings(input_ids) # (batch ,sequence_length , hidden_size)
     
        if mode == 'query':
            # ent_embedding.shape[0]  batch
            # ent_embedding.shape[2]  hidden_size
            # print(self.prompt)
            prompt_embed = self.prompt.expand(ent_embedding.shape[0], self.prompt_length, ent_embedding.shape[2]).to(self.devices)# (batch , prompt_length , hidden_size)
            inputs_embeds = torch.cat([ent_embedding[:,:-2,:], prompt_embed, ent_embedding[:,-2:,:]],dim=1)  # (batch , ent_h + prompt_length + （mask+END） , hidden_size) 
        else:
            inputs_embeds = ent_embedding # (batch ,sequence_length , hidden_size)
        # print(inputs_embeds.shape)
        # print(self.model) #BartModel
        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=inputs_embeds
        )
        # print( outputs[0].shape)#last_hidden_state (batch_size, sequence_length, hidden_size))
        return outputs
        
