from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from Siamese import Siamese
from tqdm import tqdm
import time



#This class is to actually execute all of our training functionality
class Trainer:
    def __init__(self, backbone, tokenizer, topics, evidences, procons, labels,lrate=1e-4, device='cuda:0',
                 batch_size=16, shuffle=True, num_workers=0, precomputed_dataset=None):
        self.backbone = backbone
        self.device = device

        if precomputed_dataset is not None:
            self.dataset = precomputed_dataset
        else:
            self.dataset = MultiLabelDataset(tokenizer, topics, evidences, procons, labels,
                                         device=device)
        self.dataloader = DataLoader(self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        self.model = Siamese(backbone)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=lrate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, epochs, savepath):
        for epoch in range(epochs):
            self.model.train()
            t0 = time.time()
            running_loss = 0
            for _, data in tqdm(enumerate(self.dataloader)):
                #print("dataloader elapsed time: {0}".format(time.time()-t0))
                t0 = time.time()
                inp1, inp2, targets = data
                outputs = self.model(inp1,inp2)
                #print("forward elapsed time: {0}".format(time.time()-t0))
                t0 = time.time()

                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                running_loss+=loss.item()
                if (_+1) % 100 == 0:
                    print(f'Epoch: {epoch}, Loss:  {running_loss/100}')
                    running_loss = 0

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print("backward elapsed time: {0}".format(time.time()-t0))
                t0 = time.time()
            torch.save(self.backbone,open(savepath,'wb'))


class MultiLabelDataset(Dataset): #dataset should return something that can be fed into the Siamese class

    def __init__(self, tokenizer, topics, evidences, procons, labels,device='cuda:0'):
        self.tokenizer = tokenizer
        self.topics = topics
        self.evidences = evidences
        self.procons = procons
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.procons)

    def __getitem__(self, index):
        ev1, ev2 = self.evidences[index]
        topic = self.topics[index]
        procon1, procon2 = self.procons[index]
        procon1, procon2 = torch.tensor(procon1).to(self.device), torch.tensor(procon2).to(self.device)
        tokenized_ev1 = self.tokenizer.tokenize(ev1) #already tensor
        tokenized_ev2 = self.tokenizer.tokenize(ev2) #already tensor
        tokenized_topic = self.tokenizer.tokenize(topic) #already tensor
        label = torch.tensor(self.labels[index],dtype=torch.int64).to(self.device)

        return [*tokenized_ev1, *tokenized_topic, procon1], [*tokenized_ev2, *tokenized_topic, procon2], label
        #hopefully collate_fn will work here...