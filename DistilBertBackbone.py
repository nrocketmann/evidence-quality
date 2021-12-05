from transformers import ElectraModel, ElectraTokenizerFast
from torch import nn
from Tokenizer import Tokenizer
import torch

#model class for DistilBERT backbone, with linear layers on top to predict output
#It's "simple" because we are just encoding the topic with the same model and concatenating
class DistilBERTSimple(nn.Module):
    def __init__(self, hidden_layers = [512], hidden_activation=nn.ReLU, dropout_chance = .1,num_outputs = 1, device='cuda:0'):
        super(DistilBERTSimple, self).__init__()
        self.l1 = ElectraModel.from_pretrained('google/electra-small-discriminator') #I'm not using any config here, just default
        self.l2 = ElectraModel.from_pretrained('google/electra-small-discriminator')
        #we might want to change sequence length later?? I think 512 is pretty long, but it should pad and stuff making it ok
        self.l1.requires_grad_(True)
        self.l2.requires_grad_(True)

        self.hidden_layers = []
        if dropout_chance>0:
            self.hidden_layers.append(nn.Dropout(dropout_chance))

        #add hidden layers
        last_size = 256*2# + 1 #BERT output size is 768, electra is 256
        for size in hidden_layers:
            self.hidden_layers.append(nn.Linear(last_size, size).to(device))
            self.hidden_layers.append(hidden_activation())
            self.hidden_layers.append(nn.Dropout(dropout_chance))
            last_size = size

        #add final linear layer
        self.final_layer = nn.Linear(last_size, num_outputs)
        self.dropout3 = nn.Dropout(.2)

    def forward(self, input_ids_evidence, attention_mask_evidence, input_ids_topic, attention_mask_topic, procon):
        h_ev = self.l1(input_ids=input_ids_evidence, attention_mask=attention_mask_evidence)
        h_topic = self.l2(input_ids=input_ids_topic, attention_mask=attention_mask_topic)
        weird_procon = (2*procon - 1).view(-1,1)

        #h = torch.cat([h_ev[0][:,0], h_topic[0][:,0], procon.view(-1,1)],dim=-1)
        h = torch.cat([h_ev[0][:, 0], h_topic[0][:, 0] * weird_procon],dim=-1)
        h = self.dropout3(h)
        for layer in self.hidden_layers:
            h = layer(h)
        output = self.final_layer(h)
        return output

    def parameters(self):
        for module in [self.l1, self.final_layer] + self.hidden_layers:
            for param in module.parameters():
                yield param

class DistilBERTAttention(nn.Module):
    def __init__(self, hidden_layers = [512], hidden_activation=nn.ReLU, dropout_chance = .1,num_outputs = 1, device='cuda:0'):
        super(DistilBERTAttention, self).__init__()
        self.l1 = ElectraModel.from_pretrained('google/electra-small-discriminator') #I'm not using any config here, just default
        self.l2 = ElectraModel.from_pretrained('google/electra-small-discriminator')
        #we might want to change sequence length later?? I think 512 is pretty long, but it should pad and stuff making it ok
        self.l1.requires_grad_(True)
        self.l2.requires_grad_(True)

        HSIZE = 128
        HEADS = 4
        self.attention_layer = nn.MultiheadAttention(HSIZE, HEADS, batch_first=True, dropout=.15)
        self.attention_layer2 = nn.MultiheadAttention(HSIZE, HEADS, batch_first=True, dropout=.15)
        self.keyW = nn.Linear(HSIZE * 2, HSIZE)
        self.valueW = nn.Linear(HSIZE * 2, HSIZE)
        self.queryW = nn.Linear(HSIZE * 2, HSIZE)
        self.keyW2 = nn.Linear(HSIZE, HSIZE)
        self.valueW2 = nn.Linear(HSIZE, HSIZE)
        self.queryW2 = nn.Linear(HSIZE, HSIZE)
        self.output_hidden = nn.Linear(HSIZE, 128)

        #add final linear layer
        self.final_layer = nn.Linear(128, num_outputs)
        self.dropout3 = nn.Dropout(.2)

    def forward(self, input_ids_evidence, attention_mask_evidence, input_ids_topic, attention_mask_topic, procon):
        h_ev = self.l1(input_ids=input_ids_evidence, attention_mask=attention_mask_evidence)
        h_topic = self.l2(input_ids=input_ids_topic, attention_mask=attention_mask_topic)
        weird_procon = (2*procon - 1).view(-1,1,1)
        top_hidden = self.dropout3(h_topic[0] * weird_procon)
        ev_hidden =  self.dropout3(h_ev[0])
        queries, keys, values = self.queryW(top_hidden) , self.keyW(ev_hidden), self.valueW(ev_hidden)
        hidden, attentionW = self.attention_layer(queries, keys, values)
        queries2, keys2, values2 = self.queryW2(hidden[:,0]), self.keyW2(hidden), self.valueW2(hidden)
        hidden, attentionW2 = self.attention_layer2(queries2.view(-1, 1, 128), keys2, values2)
        hidden = hidden.squeeze(1)
        out = self.final_layer(hidden)
        return out


    def parameters(self):
        for module in [self.l1, self.final_layer] + self.hidden_layers:
            for param in module.parameters():
                yield param



class DistilBERTDotProduct(nn.Module):
    def __init__(self, hidden_layers=[512], hidden_activation=nn.ReLU, dropout_chance=.1, num_outputs=1,
                 device='cuda:0'):
        super(DistilBERTDotProduct, self).__init__()
        self.l1 = ElectraModel.from_pretrained(
            'google/electra-small-discriminator')  # I'm not using any config here, just default
        # we might want to change sequence length later?? I think 512 is pretty long, but it should pad and stuff making it ok
        self.l1.requires_grad_(True)
        self.l2 = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.l2.requires_grad_(True)

        self.hidden_layers1 = []
        self.hidden_layers2 = []
        if dropout_chance > 0:
            self.hidden_layers1.append(nn.Dropout(dropout_chance))
            self.hidden_layers2.append(nn.Dropout(dropout_chance))

        # add hidden layers
        last_size = 256 # BERT output size is 768, electra is 256
        for size in hidden_layers:
            self.hidden_layers1.append(nn.Linear(last_size, size).to(device))
            self.hidden_layers1.append(hidden_activation())
            self.hidden_layers1.append(nn.Dropout(dropout_chance))
            self.hidden_layers2.append(nn.Linear(last_size, size).to(device))
            self.hidden_layers2.append(hidden_activation())
            self.hidden_layers2.append(nn.Dropout(dropout_chance))
            last_size = size

        # add final linear layer
        self.dropout3 = nn.Dropout(.3)

    def forward(self, input_ids_evidence, attention_mask_evidence, input_ids_topic, attention_mask_topic, procon):
        h_ev = self.l1(input_ids=input_ids_evidence, attention_mask=attention_mask_evidence)
        h_topic = self.l2(input_ids=input_ids_topic, attention_mask=attention_mask_topic)

        h_ev = self.dropout3(h_ev[0][:, 0])
        weird_procon = (2*procon - 1).view(-1,1)
        h_topic = self.dropout3(h_topic[0][:, 0]) * weird_procon


        for layer1, layer2 in zip(self.hidden_layers1,self.hidden_layers2):
            h_ev = layer1(h_ev)
            h_topic = layer2(h_topic)
        output = torch.sum(h_ev * h_topic,dim=-1).view(-1,1)

        return output

    def parameters(self):
        for module in [self.l1] + self.hidden_layers1 + self.hidden_layers2:
            for param in module.parameters():
                yield param

class DistilBERTTokenizer(Tokenizer):
    def __init__(self, max_len = 512, device='cuda:0'):
        super().__init__()
        self.device = device
        self.max_len=64
        self.tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')

    def tokenize(self, string):
        text = " ".join(string.split())

        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']


        return [ #return in a list format for convenience
            torch.tensor(ids, dtype=torch.long).to(self.device),
            torch.tensor(mask, dtype=torch.long).to(self.device)]
