from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizerFast
from torch import nn
from Tokenizer import Tokenizer
import torch
import re
import spacy
from spacy.tokenizer import Tokenizer as SpacyTokenizer
import gensim.downloader as api
import functools
from collections import Counter
import numpy as np
import pickle

# model class for DistilBERT backbone, with linear layers on top to predict output
# It's "simple" because we are just encoding the topic with the same model and concatenating
class LSTMBackbone(nn.Module):
    def __init__(self, num_outputs, embedding_matrix, input_length, learned_embeddings=True, device='cuda:0'):
        super(LSTMBackbone, self).__init__()
        self.num_outputs = num_outputs
        self.seq_len = input_length

        HSIZE = 128
        self.vocab_size, self.embedding_dim = embedding_matrix.shape[0], embedding_matrix.shape[1]
        self.emb_layer = nn.Embedding(self.vocab_size, self.embedding_dim,self.vocab_size-1)
        self.emb_layer.load_state_dict({'weight': embedding_matrix})
        if not learned_embeddings:
            self.emb_layer.weight.requires_grad = False

        self.lstm_ev = nn.LSTM(self.embedding_dim, HSIZE, bidirectional=True, batch_first=True)
        self.lstm_top = nn.LSTM(self.embedding_dim, HSIZE, bidirectional=True, batch_first=True)
        self.output_hidden = nn.Linear(2*2*HSIZE+1,128)
        self.output_fc = nn.Linear(128,num_outputs)


    def forward(self, evidence, evidence_lengths, topic, topic_lengths, procon):
        batch_size = procon.shape[0]
        embeddings_ev = self.emb_layer(evidence)
        embeddings_top = self.emb_layer(topic)
        embeddings_top = nn.Dropout(.3)(embeddings_top)
        embeddings_ev = nn.utils.rnn.pack_padded_sequence(embeddings_ev, evidence_lengths, batch_first=True,enforce_sorted=False)
        embeddings_top = nn.utils.rnn.pack_padded_sequence(embeddings_top, topic_lengths, batch_first=True,enforce_sorted=False)
        lstm_ev, _ = self.lstm_ev(embeddings_ev)
        lstm_top,_ = self.lstm_top(embeddings_top)

        lstm_ev, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_ev, batch_first=True)
        lstm_top, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_top, batch_first=True)
        ev_hidden = lstm_ev.contiguous()[torch.arange(batch_size),evidence_lengths-1,:]
        top_hidden = lstm_top.contiguous()[torch.arange(batch_size), topic_lengths - 1, :]
        hidden = torch.cat([ev_hidden, top_hidden, procon.view(-1,1)],dim=-1)
        hidden = nn.Dropout(.2)(hidden)

        hidden = nn.ReLU()(self.output_hidden(hidden))
        outputs = self.output_fc(hidden)
        return outputs



class DumbTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    def tokenize(self,string):
        return [string[...,:-1],string[...,-1]]


def build_dictionary(uniques):
    d = {}
    rd = {}
    for i, thing in enumerate(uniques):
        d[thing] = i
        rd[i] = thing
    return d, rd

def unkfix_toks(toks, dictionary):
    newt = []
    for t in toks:
        if t not in dictionary:
            newt.append('UNKA')
        else:
            newt.append(t)
    return newt

def make_model_datasets(topics, evidences,device):
    nlp = spacy.load("en_core_web_sm")
    tokenizer = SpacyTokenizer(nlp.vocab)
    nlp.tokenizer = tokenizer
    glovewv = api.load('glove-wiki-gigaword-50')
    tokenized_topics = [[t.text for t in nlp(topic)] for topic in topics]
    tokenized_evidences = [[t.text for t in nlp(evidence)] for evidence in evidences]
    all_tokens = functools.reduce(lambda x, y: x + y, tokenized_topics + tokenized_evidences,[])

    #build token dictionary with padding
    cnt = Counter(all_tokens)
    new_alltokens = []
    badtoks = set()

    for k,v in cnt.items():
        if v>3:
            new_alltokens.append(k)
        else:
            badtoks.add(k)

    new_alltokens.append('UNKA')
    new_alltokens.append('<PAD>')
    tokendict, tokenrdict = build_dictionary(new_alltokens)
    VOCAB_SIZE = len(tokendict)

    #build embedding matrix from GloVe
    embmatrix = np.random.normal(scale=.6,size=(VOCAB_SIZE-1,50))
    numsad = 0
    for k,v in tokendict.items():
        if k=='<PAD>':
            continue
        try:
            embmatrix[v] = glovewv[k]
        except KeyError:
            numsad+=1
    print(numsad)
    print(VOCAB_SIZE)
    embmatrix = torch.from_numpy(np.concatenate([embmatrix,np.zeros((1,50))]).astype(np.float64)) #add the zeros for padding boy

    # replace unk tokens in train_samples and pad sequences
    ev_ds_, top_ds_ = [], []
    lengths_ev, lengths_top = [], []
    SEQ_LEN = 50
    for i, topic in enumerate(tokenized_topics):
        unkfixed_top = unkfix_toks(topic, tokendict)[:SEQ_LEN]
        lengths_top.append(len(unkfixed_top))
        for _ in range(SEQ_LEN - len(unkfixed_top)):
            unkfixed_top.append('<PAD>')
        top_ds_.append(unkfixed_top)
    for i, evidence in enumerate(tokenized_evidences):
        unkfixed_ev = unkfix_toks(evidence, tokendict)[:SEQ_LEN]
        lengths_ev.append(len(unkfixed_ev))
        for _ in range(SEQ_LEN - len(unkfixed_ev)):
            unkfixed_ev.append('<PAD>')

        ev_ds_.append(unkfixed_ev)

    pickle.dump(tokendict, open('token_dictionary.pkl', 'wb'))
    ev_ds, top_ds = [], []

    # now we'll actually embed everything as integers
    for top in top_ds_:
        top_ds.append([])
        for tok in top:
            top_ds[-1].append(tokendict[tok])
    for ev in ev_ds_:
        ev_ds.append([])

        for tok in ev:
            ev_ds[-1].append(tokendict[tok])


    final_evidence = np.array(ev_ds)
    N = len(top_ds)
    final_evidence = np.stack([final_evidence[:N],final_evidence[N:]],axis=1)
    final_topics = np.array(top_ds)
    lengths_ev = np.stack([lengths_ev[:N],lengths_ev[N:]],axis=1) #shape N x 2
    final_evidence = np.concatenate([final_evidence, np.expand_dims(lengths_ev,-1)],-1)
    final_topics = np.concatenate([final_topics,np.expand_dims(np.asarray(lengths_top),1)],axis=-1)
    model = LSTMBackbone(1,embmatrix,SEQ_LEN,False,device=device)
    return model, final_evidence, final_topics, SEQ_LEN,embmatrix

def make_model_datasets_test(topics, evidences,device, dictfile):
    nlp = spacy.load("en_core_web_sm")
    tokenizer = SpacyTokenizer(nlp.vocab)
    nlp.tokenizer = tokenizer
    tokenized_topics = [[t.text for t in nlp(topic)] for topic in topics]
    tokenized_evidences = [[t.text for t in nlp(evidence)] for evidence in evidences]
    tokendict = pickle.load(open(dictfile,'rb'))

    #build token dictionary with padding
    VOCAB_SIZE = len(tokendict)

    # replace unk tokens in train_samples and pad sequences
    ev_ds_, top_ds_ = [], []
    lengths_ev, lengths_top = [], []
    SEQ_LEN = 50
    for i, topic in enumerate(tokenized_topics):
        unkfixed_top = unkfix_toks(topic, tokendict)[:SEQ_LEN]
        lengths_top.append(len(unkfixed_top))
        for _ in range(SEQ_LEN - len(unkfixed_top)):
            unkfixed_top.append('<PAD>')
        top_ds_.append(unkfixed_top)

    for i, evidence in enumerate(tokenized_evidences):
        unkfixed_ev = unkfix_toks(evidence, tokendict)[:SEQ_LEN]

        lengths_ev.append(len(unkfixed_ev))


        for _ in range(SEQ_LEN - len(unkfixed_ev)):
            unkfixed_ev.append('<PAD>')

        ev_ds_.append(unkfixed_ev)
    ev_ds, top_ds = [], []

    # now we'll actually embed everything as integers
    for top in top_ds_:
        top_ds.append([])
        for tok in top:
            top_ds[-1].append(tokendict[tok])
    for ev in ev_ds_:
        ev_ds.append([])
        for tok in ev:
            ev_ds[-1].append(tokendict[tok])

    final_evidence = np.array(ev_ds)
    N = len(top_ds)
    final_evidence = np.stack([final_evidence[:N],final_evidence[N:]],axis=1)
    final_topics = np.array(top_ds)
    lengths_ev = np.stack([lengths_ev[:N],lengths_ev[N:]],axis=1) #shape N x 2
    final_evidence = np.concatenate([final_evidence, np.expand_dims(lengths_ev,-1)],-1)
    final_topics = np.concatenate([final_topics,np.expand_dims(np.asarray(lengths_top),1)],axis=-1)
    return final_evidence, final_topics




