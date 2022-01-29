import os
import io
import requests
import numpy as np
import pandas as pd
import re
import zipfile
import random
import time
import csv
import datetime
from itertools import compress
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
                         AdamW, get_linear_schedule_with_warmup, \
                         TrainingArguments, BeamScorer, Trainer

from transformers.optimization import  Adafactor 
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import warnings

import torch
#from torch.utils.data import Dataset, random_split, DataLoader, \
                            # RandomSampler, SequentialSampler

from IPython.display import clear_output


import json


#WebNLG
import glob
import xml.etree.ElementTree as ET
import urllib.request
import zipfile


import warnings
warnings.filterwarnings('ignore')


from IPython.display import HTML, display


class Model:
    
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)

        if torch.cuda.is_available():
            self.dev = torch.device("cuda:0") 
            print("Running on the GPU")
        else:
            self.dev = torch.device("cpu")
            print("Running on the CPU")
        
        self.model.to(self.dev)
        self.optimizer = Adafactor(
            self.model.parameters(),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
    
    def train(self,train_df):
        batch_size=4
        num_of_batches=len(train_df)/batch_size
        num_of_epochs=2

        num_of_batches=int(num_of_batches)

        #Sets the module in training mode
        self.model.train()

        loss_per_10_steps=[]
        for epoch in range(1,num_of_epochs+1):
            print('Running epoch: {}'.format(epoch))
            
            running_loss=0

            #out = display(self.progress(1, num_of_batches+1), display_id=True)
            
            for i in range(num_of_batches):
                inputbatch=[]
                labelbatch=[]
                new_df=train_df[i*batch_size:i*batch_size+batch_size]

                for indx,row in new_df.iterrows():
                    input = row['input_text']+'</s>' 
                    labels = row['target_text']+'</s>'   
                    inputbatch.append(input)
                    labelbatch.append(labels)

                inputbatch=self.tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
                labelbatch=self.tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,return_tensors="pt") ["input_ids"]
                inputbatch=inputbatch.to(self.dev)
                labelbatch=labelbatch.to(self.dev)

                # clear out the gradients of all Variables 
                self.optimizer.zero_grad()

                # Forward propogation
                outputs = self.model(input_ids=inputbatch, labels=labelbatch)
                loss = outputs.loss
                loss_num=loss.item()
                logits = outputs.logits
                running_loss+=loss_num

                if i%10 ==0:      
                    loss_per_10_steps.append(loss_num)

                #out.update(self.progress(loss_num,i, num_of_batches+1))

                # calculating the gradients
                loss.backward()

                #updating the params
                self.optimizer.step()
                
            running_loss=running_loss/int(num_of_batches)
            print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))
            

    def text(self):
        self.model.eval()
        input_ids = self.tokenizer.encode("Mary | hometown | London && Mary | birthday | 1950 </s>", return_tensors="pt")  # Batch size 1
        input_ids=input_ids.to(self.dev)
        outputs = self.model.generate(input_ids)
        result = self.tokenizer.decode(outputs[0])
        return result
        

    def progress(loss,value, max=100):
        return HTML(""" Batch loss :{loss}
            <progress
                value='{value}'
                max='{max}',
                style='width: 100%'
            >
                {value}
            </progress>
        """.format(loss=loss,value=value, max=max))