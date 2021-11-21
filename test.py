from DistilBertBackbone import DistilBERTSimple, DistilBERTTokenizer
from Trainer import Trainer
import pandas as pd
import numpy as np
import torch

device = 'cpu'
batch_size=2
SAVEPATH = "model.pth"

def get_datas(df):
    evidences =  np.stack([df['evidence_1'].values, df['evidence_2'].values],axis=-1)
    procon1 = df['evidence_1_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procon2 = df['evidence_2_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procons = np.stack([procon1.values,procon2.values],axis=-1)
    topics = df['topic'].values
    targets = df['label'].values-1
    return topics, evidences, procons, targets


df = pd.read_csv('data/test.csv')
train_data = get_datas(df)

tokenizer = DistilBERTTokenizer(device=device)
backbone = torch.load(SAVEPATH)
trainer = Trainer(backbone,tokenizer,*train_data,device=device,batch_size=batch_size)

print("Model and data loaded! Beginning training")
acc = trainer.evaluate()
print(ac)