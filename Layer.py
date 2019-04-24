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

class SimpleCatTgtMasked(nn.Module):
    def __init__(self, config):
        '''
        Concatenate word embeddings and target embeddings
        '''
        super(SimpleCatTgtMasked, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.embed_num, config.embed_dim)
        #A special embedding for target
        self.target_emb = torch.rand(config.embed_dim)*0.025-0.05
        
        
        self.mask_embed = nn.Embedding(2, config.mask_dim)
        
        #positional embeddings
        n_position = 100
        self.position_enc = nn.Embedding(n_position, config.embed_dim, padding_idx=0)
        self.position_enc.weight.data = self.position_encoding_init(n_position, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        

    # input are tensors
    def forward(self, sent, mask, is_elmo=False, is_pos=False):
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
            self.target_emb = self.target_emb.cuda()
        # to embeddings
        if is_elmo:
            sent_vec = sent # batch_siz*sent_len * dim
        else:
            sent_vec = self.word_embed(sent)# batch_siz*sent_len * dim
            
            
        #Replace target with a special embedding
        for i, m in enumerate(mask):
            target_index = m.argmax()
            #print(m)
            sent_vec[i, target_index] = self.target_emb
                

        sent_vec = self.dropout(sent_vec)
        
        #positional embeddings
        if is_pos:
            batch_size, max_len, _ = sent_vec.size()
            pos = torch.arange(0, max_len)
            if self.config.if_gpu:pos = pos.cuda()
            pos = pos.expand(batch_size, max_len)
            pos_vec = self.position_enc(pos)
            sent_vec += pos_vec
            
        mask_vec = self.mask_embed(mask) # batch_size*max_len* dim
        #print(mask_vec.size())
        
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
            self.position_enc.requires_grad = self.config.if_update_embed
            self.target_emb.requires_grad = True
            print('embeddings loaded')
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()
        
    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])


        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        pos_emb = torch.from_numpy(position_enc).type(torch.FloatTensor)
        if self.config.if_gpu: pos_emb = pos_emb.cuda()
        return pos_emb
    
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
        
        #positional embeddings
        n_position = 100
        self.position_enc = nn.Embedding(n_position, config.embed_dim, padding_idx=0)
        self.position_enc.weight.data = self.position_encoding_init(n_position, config.embed_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.senti_embed = nn.Embedding(config.embed_num, 50)

    # input are tensors
    def forward(self, sent, mask, is_elmo=False, is_pos=False):
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
        if is_elmo:
            sent_vec = sent # batch_siz*sent_len * dim
        else:
            sent_vec = self.word_embed(sent)# batch_siz*sent_len * dim
        
        sent_vec = self.dropout(sent_vec)
        
        ############
#         #Add sentiment-specific embeddings
#         senti_vec = self.senti_embed(sent)
#         sent_vec = torch.cat([sent_vec, senti_vec], 2)
                             
                             
        
        
        #positional embeddings
        if is_pos:
            batch_size, max_len, _ = sent_vec.size()
            pos = torch.arange(0, max_len)
            if self.config.if_gpu:pos = pos.cuda()
            pos = pos.expand(batch_size, max_len)
            pos_vec = self.position_enc(pos)
            sent_vec += pos_vec
            
        mask_vec = self.mask_embed(mask) # batch_size*max_len* dim
        #print(mask_vec.size())
        
        #Concatenation
        sent_vec = torch.cat([sent_vec, mask_vec], 2)

        # for test
        return sent_vec

    def load_vector(self):
        '''
        Load pre-savedd word embeddings
        '''
        with open(self.config.embed_path, 'rb') as f:
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.word_embed.weight.requires_grad = self.config.if_update_embed
            self.position_enc.requires_grad = self.config.if_update_embed
            print('embeddings loaded')
            
    def load_sswu_dict(self):
        '''
        Load pre-saved sentiment-specific word embeddings
        '''
        path = 'data/tweets/vocab/sswe-u.pkl'
        with open(path, 'rb') as f:
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(path, vectors.shape))
            self.senti_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.senti_embed.weight.requires_grad = self.config.if_update_embed
            print('Sentiment specific embedding loaded')

        
    
    def reset_binary(self):
        self.mask_embed.weight.data[0].zero_()
        
    def position_encoding_init(self, n_position, emb_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])


        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        pos_emb = torch.from_numpy(position_enc).type(torch.FloatTensor)
        if self.config.if_gpu: pos_emb = pos_emb.cuda()
        return pos_emb


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
        
        
def get_dyt_emb(file = '../data/sswe-u.txt'):
    word_emb = {}
    with open(file) as fi:
        for line in fi:
            items = line.split()
            word_emb[items[0]] = np.array(items[1:], dtype=np.float32)
    return word_emb
