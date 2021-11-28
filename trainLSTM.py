from LSTMBackbone import *
from Trainer import Trainer
import pandas as pd
import numpy as np
import torch

device = 'cpu'
lrate = 1e-4
epochs = 10
batch_size=16
SAVEPATH = "modelLSTM.pth"

def get_datas(df):
    evidences =  np.concatenate([df['evidence_1'].values, df['evidence_2'].values],axis=0)
    procon1 = df['evidence_1_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procon2 = df['evidence_2_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procons = np.stack([procon1.values,procon2.values],axis=-1)
    topics = df['topic'].values
    targets = df['label'].values-1
    return topics, evidences, procons, targets



df = pd.read_csv('data/train.csv')
topics, evidences, procons, targets = get_datas(df)
print(evidences.shape)
backbone, evidences, topics = make_model_datasets(topics,evidences,device)

iter_per_epoch = len(topics)//batch_size
print("Iter per epoch: " + str(iter_per_epoch))

tokenizer = DumbTokenizer()
trainer = Trainer(backbone,tokenizer,topics,evidences,procons,targets,device=device,batch_size=batch_size,lrate=lrate)

print("Model and data loaded! Beginning training")
trainer.train(epochs,SAVEPATH)
torch.save(backbone, open(SAVEPATH,'wb'))
