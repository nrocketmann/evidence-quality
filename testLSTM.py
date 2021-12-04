from LSTMBackbone import LSTMBackbone,DumbTokenizer,make_model_datasets_test
from Trainer import Trainer
import pandas as pd
import numpy as np
import torch
from torch import nn

device = 'cpu'
batch_size=16
SAVEPATH = "modelLSTMVanilla.pth"

def get_datas(df):
    #df = df[-200:]
    evidences =  np.concatenate([df['evidence_1'].values, df['evidence_2'].values],axis=0)
    procon1 = df['evidence_1_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procon2 = df['evidence_2_stance'].apply(lambda x: 0 if x=="CON" else 1)
    procons = np.stack([procon1.values,procon2.values],axis=-1)
    topics = df['topic'].values
    targets = df['label'].values-1
    return topics, evidences, procons, targets

backbone = torch.load(SAVEPATH)

df = pd.read_csv('data/test.csv')
topics, evidences, procons, targets = get_datas(df)
evidences, topics = make_model_datasets_test(topics,evidences,device,'token_dictionary.pkl')

tokenizer = DumbTokenizer()
trainer = Trainer(backbone,tokenizer,topics,evidences,procons,targets,device=device,batch_size=batch_size,shuffle=False)

print("Model and data loaded! Beginning testing")
acc = trainer.evaluate()
print(acc)