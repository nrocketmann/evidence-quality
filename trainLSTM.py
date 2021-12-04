from LSTMBackbone import *
from Trainer import Trainer
import pandas as pd
import numpy as np
import torch

device = 'cuda:0'
lrate = 1e-3
epochs = 50
batch_size=32
SAVEPATH = "modelLSTMAttention.pth"
cache_dataset="lstmcache.pkl"
load_cache = True

use_attention = True
learned_embeddings = False
num_outputs = 1

#1 output, no attention, no learned embeddings: 67-69 test accuracy

def get_datas(df):
    evidences =  np.concatenate([df['evidence_1'].values, df['evidence_2'].values],axis=0)
    procon1 = df['evidence_1_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procon2 = df['evidence_2_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procons = np.stack([procon1.values,procon2.values],axis=-1)
    topics = df['topic'].values
    targets = df['label'].values-1
    return topics, evidences, procons, targets

if load_cache:
    loaded = pickle.load(open(cache_dataset,'rb'))
    evidences = loaded["evidence"]
    topics = loaded["topic"]
    procons = loaded["procons"]
    targets = loaded["targets"]
    seq_len = loaded['seq_len']
    embmatrix = loaded['embmatrix']
    backbone = LSTMBackbone(num_outputs,embmatrix,seq_len,learned_embeddings,use_attention,device=device)
else:
    df = pd.read_csv('data/train.csv')
    topics, evidences, procons, targets = get_datas(df)
    backbone, evidences, topics, seq_len, embmatrix = make_model_datasets(topics,evidences,device,num_outputs,learned_embeddings,use_attention)
    if cache_dataset!="":
        pickle.dump({"evidence":evidences,"topic":topics,"procons":procons,"targets":targets,'seq_len':seq_len, 'embmatrix':embmatrix},open(cache_dataset,'wb'))

iter_per_epoch = len(topics)//batch_size
print("Iter per epoch: " + str(iter_per_epoch))

tokenizer = DumbTokenizer(device)
trainer = Trainer(backbone,tokenizer,topics,evidences,procons,targets,device=device,batch_size=batch_size,lrate=lrate)

print("Model and data loaded! Beginning training")
trainer.train(epochs,SAVEPATH)
