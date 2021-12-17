from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from Siamese import Siamese
from tqdm import tqdm
import time



#This class is to actually execute all of our training functionality
class Trainer:
    def __init__(self, backbone, tokenizer, topics, evidences, procons, labels,lrate=1e-4, device='cuda:0',
                 batch_size=16, shuffle=True, num_workers=0, valsize = 253+170):
        self.backbone = backbone
        self.device = device

        if valsize!=0:
            self.traindataset = MultiLabelDataset(tokenizer, topics[:-valsize], evidences[:-valsize], procons[:-valsize], labels[:-valsize],
                                         device=device)
            self.traindataloader = DataLoader(self.traindataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)

            self.valdataset = MultiLabelDataset(tokenizer, topics[-valsize:], evidences[-valsize:], procons[-valsize:], labels[-valsize:],
                                         device=device)
            self.valdataloader = DataLoader(self.valdataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        else:
            self.dataset = MultiLabelDataset(tokenizer, topics, evidences, procons, labels,
                                         device=device)
            self.dataloader = DataLoader(self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        self.model = Siamese(backbone).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=lrate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, epochs, savepath):
        valaccs = []
        for epoch in range(epochs):
            self.model.train()
            t0 = time.time()
            running_loss = 0
            for _, data in tqdm(enumerate(self.traindataloader)):
                self.model.train()
                self.model.backbone.train()
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
                    print('Epoch: {0}, Loss:  {1}'.format(epoch,running_loss/100))
                    running_loss = 0

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 1.0)
                self.optimizer.step()
                #print("backward elapsed time: {0}".format(time.time()-t0))
                t0 = time.time()
            torch.save(self.backbone,open(savepath,'wb'))
            valaccs.append(self.valset())
            print("Validation Accuracy: {0}".format(valaccs[-1]))
            if len(valaccs)>10 and valaccs[-10]>=valaccs[-1]:
                print("Early stopping!")
                #break

    def valset(self):
        self.model.eval()
        correct_score = 0
        total_score = 0
        for _, data in tqdm(enumerate(self.valdataloader)):
            self.model.backbone.eval()
            self.model.eval()
            inp1, inp2, targets = data
            outputs = self.model(inp1, inp2)
            preds = torch.argmax(outputs,dim=-1)
            for p, t in zip(preds, targets):
                if p==t:
                    correct_score+=1
                total_score+=1
        return correct_score/total_score
    def evaluate(self):
        self.model.eval()
        correct_score = 0
        total_score = 0
        weight_list = []
        for _, data in tqdm(enumerate(self.dataloader)):
            self.model.backbone.eval()
            self.model.eval()
            inp1, inp2, targets = data
            outputs, aw1, aw2 = self.model(inp1, inp2, return_weight=True)
            preds = torch.argmax(outputs,dim=-1)
            for p, t in zip(preds, targets):
                if p==t:
                    correct_score+=1
                total_score+=1
            weight_list.append((aw1, aw2))
        print(total_score)
        return correct_score/total_score, weight_list



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