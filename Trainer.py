from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from Siamese import Siamese
from tqdm import tqdm



#This class is to actually execute all of our training functionality
class Trainer:
    def __init__(self, backbone, tokenizer, topics, evidences, procons, labels,lrate=1e-4, device='cuda:0',
                 batch_size=16, shuffle=True, num_workers=0):
        self.backbone = backbone
        self.device = device

        self.dataset = MultiLabelDataset(tokenizer, topics, evidences, procons, labels,
                                         device=device)
        self.dataloader = DataLoader(self.dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        self.model = Siamese(backbone)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=lrate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for _, data in tqdm(enumerate(self.dataloader)):
                inp1, inp2, targets = data
                outputs = self.model(inp1,inp2)

                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                if (_+1) % 100 == 0:
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



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
        procon = torch.tensor(self.procons[index]).to(self.device)
        tokenized_ev1 = self.tokenizer.tokenize(ev1) #already tensor
        tokenized_ev2 = self.tokenizer.tokenize(ev2) #already tensor
        tokenized_topic = self.tokenizer.tokenize(topic) #already tensor

        return [tokenized_ev1, tokenized_topic, procon], [tokenized_ev2, tokenized_topic, procon], self.labels[index] #hopefully collate_fn will work here...