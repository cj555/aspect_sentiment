from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import pdb
import pickle
import numpy as np
import math
from torch.nn import utils as nn_utils
from util import *

def sent_split(sent):
    words = []
    sent = nlp(sent)
    for w in sent:
        words.append(w.text.lower())
    return words

def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

class MLSTM(nn.Module):
    def __init__(self, config):
        super(MLSTM, self).__init__()
        self.config = config

        self.rnn = nn.LSTM(config.embed_dim + config.mask_dim, int(config.l_hidden_size / 2), batch_first=True, num_layers = int(config.l_num_layers / 2),
            bidirectional=True, dropout=config.l_dropout)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        #FIXIT: doesn't have batch
        #Sort the lengths
        # seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        # feats = feats[perm_idx]
        #feats = feats.unsqueeze(0)
        pack = nn_utils.rnn.pack_padded_sequence(feats, 
                                                 seq_lengths, batch_first=True)
        
        
        #assert self.batch_size == batch_size
        lstm_out, _ = self.rnn(pack)
        #lstm_out, (hid_states, cell_states) = self.rnn(feats)

        #Unpack the tensor, get the output for varied-size sentences
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        #FIXIT: for batch
        #lstm_out = lstm_out.squeeze(0)
        # batch * sent_l * 2 * hidden_states 
        return unpacked

# input layer for 14
class SimpleCat(nn.Module):
    def __init__(self, config):
        '''
        Concatenate word embeddings and target embeddings
        '''
        super(SimpleCat, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)
        self.dropout = nn.Dropout(config.dropout)

    # input are tensors
    def forward(self, sent, mask, is_elmo=False):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len, emb_dim)
        mask: tensor, shape(batch_size, max_len)
        '''
        #Modified by Richard Sun
        #Use ELmo embedding, note we need padding
        sent = Variable(sent)
        mask = Variable(mask)

        #Use GloVe embedding
        if self.config.if_gpu:  
            sent, mask = sent.cuda(), mask.cuda()
        # to embeddings
        sent_vec = sent # batch_siz*sent_len * dim
        if is_elmo:
            sent_vec = sent # batch_siz*sent_len * dim
        else:
            sent_vec = self.word_embed(sent)# batch_siz*sent_len * dim
            
        mask_vec = self.mask_embed(mask) # batch_size*max_len* dim
        #print(mask_vec.size())
        
        sent_vec = self.dropout(sent_vec)
        #Concatenation
        sent_vec = torch.cat([sent_vec, mask_vec], 2)

        # for test
        return sent_vec

    def load_vector(self):
        with open(self.config.embed_path, 'rb') as f:
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.word_embed.weight.requires_grad = self.config.if_update_embed
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()


    # input layer for 14
class GloveMaskCat(nn.Module):
    def __init__(self, config):
        super(GloveMaskCat, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)

        self.dropout = nn.Dropout(config.dropout)

    # input are tensors
    def forward(self, sent, mask, is_avg=True):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len)
        mask: tensor, shape(batch_size, max_len)
        '''
        #Modified by Richard Sun
        #Use ELmo embedding, note we need padding

        #Use GloVe embedding
        if self.config.if_gpu:  
            sent, mask = sent.cuda(), mask.cuda()
        # to embeddings
        sent_vec = self.word_embed(sent) # batch_siz*sent_len * dim
        #Concatenate each word embedding with target word embeddings' average
        batch_size, max_len = sent.size()
        #Repeat the mask
        mask = mask.type_as(sent_vec)
        mask = mask.expand(self.config.embed_dim, batch_size, max_len)
        mask = mask.transpose(0, 1).transpose(1, 2)#The same size as sentence vector
        target_emb = sent_vec * mask
        target_emb_avg = torch.sum(target_emb, 1)/torch.sum(mask, 1)#Batch_size*embedding
        #Expand dimension for concatenation
        target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, self.config.embed_dim)
        target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)#Batch_size*max_len*embedding
        if is_avg:
            target = target_emb_avg
        else:
            target = target_emb

        #sent_vec = self.dropout(sent_vec)
        #Concatenation
        #sent_target_concat = torch.cat([sent_vec, target_emb_avg_exp], 2)

        # for test
        return sent_vec, target

    def load_vector(self):
        with open(self.config.embed_path, 'rb') as f:
            #vectors = pickle.load(f, encoding='bytes')
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.word_embed.weight.requires_grad = self.config.if_update_embed
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()


# input layer for 14
class ContextTargetCat(nn.Module):
    def __init__(self, config):
        super(ContextTargetCat, self).__init__()
        '''
        This class is to concatenate the context and target embeddings
        '''
        self.config = config
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)

        self.dropout = nn.Dropout(config.dropout)

    # input are tensors
    def forward(self, sent, mask, is_concat=True):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len, dim) elmo
        mask: tensor, shape(batch_size, max_len)
        '''
        #Modified by Richard Sun
        #Use ELmo embedding, note we need padding


        if self.config.if_gpu:  
            sent, mask = sent.cuda(), mask.cuda()
        # # to embeddings
        
        batch_size, max_len, _ = sent.size()
        sent_vec = sent
        if is_concat:
            ## concatenate each word embedding with mask embedding
            mask_emb = self.mask_embed(mask)
            sent_target_concat = torch.cat([sent_vec, mask_emb], 2)
            sent_target = sent_target_concat
        else:#Add each word embedding with target word embeddings' average
            #Repeat the mask
            mask = mask.type_as(sent_vec)
            mask = mask.expand(self.config.embed_dim, batch_size, max_len)
            mask = mask.transpose(0, 1).transpose(1, 2)#The same size as sentence vector
            target_emb = sent_vec * mask
            target_emb_avg = torch.sum(target_emb, 1)/torch.sum(mask, 1)#Batch_size*embedding
            #Expand dimension for concatenation
            target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, self.config.embed_dim)
            target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)#Batch_size*max_len*embedding
            sent_target = (sent_vec + target_emb_avg_exp)/2


        
        #sent_vec = self.dropout(sent_vec)

        # for test
        return sent_target

    def load_vector(self):
        with open(self.config.embed_path, 'rb') as f:
            #vectors = pickle.load(f, encoding='bytes')
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.word_embed.weight.requires_grad = False
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()
