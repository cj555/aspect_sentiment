import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb
import pickle
from CRF import LinearCRF
import torch.nn.init as init
import pdb
import pickle
import numpy as np
# from Layer import SimpleCat
from torch.nn import utils as nn_utils
from util import *


def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config

        self.rnn = nn.LSTM(config.embed_dim + config.mask_dim, config.l_hidden_size, batch_first=True,
                           num_layers=int(config.l_num_layers / 2),
                           bidirectional=False, dropout=config.l_dropout)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        pack = nn_utils.rnn.pack_padded_sequence(feats,
                                                 seq_lengths, batch_first=True)

        # batch_size*max_len*hidden_dim
        lstm_out, (h, c) = self.rnn(pack)
        # batch_size*emb_dim
        return h[0]


class SimpleCat(nn.Module):
    def __init__(self, config):
        '''
        Concatenate word embeddings and target embeddings
        '''
        super(SimpleCat, self).__init__()
        self.config = config
        with open(self.config.embed_path, 'rb') as f:
            vectors = pickle.load(f).local_emb

        self.local_emb = vectors
        self.word_embed = nn.Embedding(vectors.shape[0], vectors.shape[1])
        self.mask_embed = nn.Embedding(2, 50)

        # positional embeddings
        # n_position = 100
        # self.position_enc = nn.Embedding(n_position, config.embed_dim, padding_idx=0)
        # # self.position_enc.weight.data = self.position_encoding_init(n_position, config.embed_dim)
        #
        self.dropout = nn.Dropout(0.1)
        #
        # self.senti_embed = nn.Embedding(config.embed_num, 50)

    # input are tensors
    def forward(self, sent, mask, target, is_elmo=False, is_pos=False):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len, emb_dim)
        mask: tensor, shape(batch_size, max_len)
        '''
        # Modified by Richard Sun
        # Use ELmo embedding, note we need padding
        # sent = Variable(sent)
        # mask = Variable(mask)
        # target = Variable(target)

        # Use GloVe embedding
        if self.config.if_gpu:
            sent, mask, target = sent.cuda(), mask.cuda(), target.cuda()
            # sent, mask= sent.cuda(), mask.cuda()
        # to embeddings
        if is_elmo:
            sent_vec = sent  # batch_siz*sent_len * dim
        else:
            sent_vec = self.word_embed(sent)  # batch_siz*sent_len * dim

        # target_vec = self.word_embed(target)
        sent_vec = self.dropout(sent_vec)
        target_vec = self.word_embed(target)
        # target_vec = self.dropout(target_vec)
        ############
        #         #Add sentiment-specific embeddings
        #         senti_vec = self.senti_embed(sent)
        #         sent_vec = torch.cat([sent_vec, senti_vec], 2)




        # positional embeddings
        if is_pos:
            batch_size, max_len, _ = sent_vec.size()
            pos = torch.arange(0, max_len)
            if self.config.if_gpu: pos = pos.cuda()
            pos = pos.expand(batch_size, max_len)
            pos_vec = self.position_enc(pos)
            sent_vec += pos_vec

        mask_vec = self.mask_embed(mask)  # batch_size*max_len* dim
        # print(mask_vec.size())

        # Concatenation
        sent_vec = torch.cat([sent_vec, mask_vec], 2)

        # for test
        return sent_vec, target_vec

    def load_vector(self):
        '''
        Load pre-savedd word embeddings
        '''
        with open(self.config.embed_path, 'rb') as f:
            vectors = pickle.load(f).local_emb

            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            # vectors = self.config.local_emb
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            # self.word_embed.weight.requires_grad = self.config.if_update_embed
            self.word_embed.weight.requires_grad = False

            # self.position_enc.requires_grad = self.config.if_update_embed
            print('embeddings loaded')


class biLSTM(nn.Module):
    def __init__(self, config):
        super(biLSTM, self).__init__()
        self.config = config

        self.rnn = nn.LSTM(config.embed_dim + config.mask_dim, int(config.l_hidden_size / 2), batch_first=True,
                           num_layers=int(config.l_num_layers / 2),
                           bidirectional=True, dropout=config.l_dropout)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        pack = nn_utils.rnn.pack_padded_sequence(feats,
                                                 seq_lengths, batch_first=True)

        # batch_size*max_len*hidden_dim
        lstm_out, _ = self.rnn(pack)
        # Unpack the tensor, get the output for varied-size sentences
        # padding with zeros
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # batch * sent_l * 2 * hidden_states
        return unpacked


class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, config):
        '''
        In this model, only context words are processed by gated CNN, target is average word embeddings
        '''
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.config = config

        # V = config.embed_num
        D = 300 + 50  # word embedding + mask embedding
        C = 3  # config.class_num

        Co = 128  # kernel numbers
        Ks = [2, 3, 4]  # kernel filter size

        # self.embed = nn.Embedding(V, D)
        # self.embed.weight = nn.Parameter(args.embedding, requires_grad=True)

        # self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
        # self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])

        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc_aspect = nn.Linear(D - 50, Co)
        self.cat_layer = SimpleCat(config)
        self.cat_layer.load_vector()

    def compute_score(self, sents, targets, lens):
        '''
        inputs are list of list for the convenince of top CRF
        Args:
        sent: a list of sentences， batch_size*max_len*(2*emb_dim)
        target: a list of target embedding for each sentence, batch_size*emb_dim
        label: a list labels
        '''

        # Get the target embedding
        batch_size, sent_len, dim = sents.size()
        # Mask the padding embeddings
        pack = nn_utils.rnn.pack_padded_sequence(sents,
                                                 lens, batch_first=True)
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(pack, batch_first=True)
        # Conv input: batch_size * emb_dim * max_len
        # Conv output: batch_size * out_dim * (max_len-k+1)
        x = [torch.tanh(conv(unpacked.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [torch.relu(conv(unpacked.transpose(1, 2)) + self.fc_aspect(targets).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]  # batch_size * out_dim * (max_len-k+1) * len(filters)

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,out_dim), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        sents_vec = torch.cat(x0, 1)  # N*(3*out_dim)
        # logit = self.fc1(x0)  # (N,C)

        # Dropout
        # if self.training:
        #     sents_vec = F.dropout(sents_vec, 0.2)

        output = self.fc1(sents_vec)  # Bach_size*label_size

        scores = F.log_softmax(output, dim=1)  # Batch_size*label_size
        return scores, output

    def forward(self, sent, masks, target, label, lens):
        # Sent emb_dim + 50
        sent, target = self.cat_layer(sent, masks, target)  # mask embedding

        target = target.sum(1) / target.size(1)
        # sent = F.dropout(sent, p=0.5, training=self.training)
        scores, output = self.compute_score(sent, target, lens)
        loss = nn.NLLLoss()
        # cls_loss = -1 * torch.log(scores[label])
        # cls_loss = F.cross_entropy(output, label)
        cls_loss = loss(scores, label)
        # print('Transition', pena)
        return cls_loss

    def predict(self, sent, masks, target, sent_len):
        sent, target = self.cat_layer(sent, masks, target)
        target = target.sum(1) / target.size(1)
        scores, output = self.compute_score(sent, target, sent_len)
        _, pred_label = scores.max(1)  # Find the max label in the 2nd dimension

        # Modified by Richard Sun
        return pred_label


def convert_mask_index(masks):
    '''
    Find the indice of none zeros values in masks, namely the target indice
    '''
    target_indice = []
    max_len = 0
    try:
        for mask in masks:
            indice = torch.nonzero(mask == 1).squeeze(1).cpu().numpy()
            if max_len < len(indice):
                max_len = len(indice)
            target_indice.append(indice)
    except:
        print('Mask Data Error')
        print(mask)
    return target_indice, max_len
