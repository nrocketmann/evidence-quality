from DistilBertBackbone import DistilBERTSimple, DistilBERTTokenizer
from Trainer import Trainer
import pandas as pd
import numpy as np
import torch

device = 'cuda:0'
lrate = 1e-4
epochs = 10
batch_size=4
SAVEPATH = "model.pth"

def get_datas(df):
    evidences =  np.stack([df['evidence_1'].values, df['evidence_2'].values],axis=-1)
    procon1 = df['evidence_1_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procon2 = df['evidence_2_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procons = np.stack([procon1.values,procon2.values],axis=-1)
    topics = df['topic'].values
    targets = df['label'].values-1
    return topics, evidences, procons, targets


df = pd.read_csv('data/train.csv')
train_data = get_datas(df)
iter_per_epoch = len(train_data[0])//batch_size
print("Iter per epoch: " + str(iter_per_epoch))

tokenizer = DistilBERTTokenizer(device=device)
backbone = DistilBERTSimple(hidden_layers=[]).to(device)
trainer = Trainer(backbone,tokenizer,*train_data,device=device,batch_size=batch_size)

print("Model and data loaded! Beginning training")
trainer.train(epochs)
torch.save(backbone, open("backbone.pth",'wb'))
