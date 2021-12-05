from DistilBertBackbone import DistilBERTSimple, DistilBERTTokenizer, DistilBERTDotProduct, DistilBERTAttention
from Trainer import Trainer
import pandas as pd
import numpy as np
import torch

device = 'cpu'
lrate = 1e-3
epochs = 20
batch_size=2
SAVEPATH = "modelBERT.pth"

def get_datas(df):
    df = df.sort_values("topic")
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
backbone = DistilBERTAttention(hidden_layers=[512,128], device=device).to(device)
trainer = Trainer(backbone,tokenizer,*train_data,device=device,batch_size=batch_size)

print("Model and data loaded! Beginning training")
trainer.train(epochs,SAVEPATH)
torch.save(backbone, open(SAVEPATH,'wb'))
