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

# input layer for 14
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
    def forward(self, sent, mask, is_elmo=False):
        '''
        Args:
        sent: tensor, shape(batch_size, max_len, emb_dim)
        mask: tensor, shape(batch_size, max_len)
        '''
        # Modified by Richard Sun
        # Use ELmo embedding, note we need padding
        sent = Variable(sent)
        mask = Variable(mask)

        # Use GloVe embedding
        if self.config.if_gpu:
            sent, mask = sent.cuda(), mask.cuda()
        # to embeddings
        sent_vec = sent  # batch_siz*sent_len * dim
        if is_elmo:
            sent_vec = sent  # batch_siz*sent_len * dim
        else:
            sent_vec = self.word_embed(sent)  # batch_siz*sent_len * dim

        mask_vec = self.mask_embed(mask)  # batch_size*max_len* dim
        # print(mask_vec.size())

        sent_vec = self.dropout(sent_vec)
        # Concatenation
        sent_vec = torch.cat([sent_vec, mask_vec], 2)

        # for test
        return sent_vec

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


# consits of three components
class CRFAspectSent(nn.Module):
    def __init__(self, config):
        '''
        LSTM+Aspect
        '''
        super(CRFAspectSent, self).__init__()
        self.config = config

        self.bilstm = biLSTM(config)
        self.feat2tri = nn.Linear(config.l_hidden_size, 2)
        self.inter_crf = LinearCRF(config)
        self.feat2label = nn.Linear(config.l_hidden_size, 3)

        self.loss = nn.NLLLoss()
        self.cat_layer = SimpleCat(config)
        self.cat_layer.load_vector()
        # Modified by Richard Sun

    def compute_scores(self, sents, masks, lens):
        '''
        Args:
        sents: batch_size*max_len*word_dim
        masks: batch_size*max_len
        lens: batch_size
        '''

        # Context embeddings
        context = self.bilstm(sents, lens)  # Batch_size*sent_len*hidden_dim
        batch_size, max_len, hidden_dim = context.size()
        # Target embeddings
        # Find target indices, a list of indices
        target_indices, target_max_len = convert_mask_index(masks)

        # Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(hidden_dim, batch_size, max_len).transpose(0, 1).transpose(1, 2)
        target_emb = masks * context

        target_emb_avg = torch.sum(target_emb, 1) / torch.sum(masks, 1)  # Batch_size*embedding
        # Expand dimension for concatenation
        target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, hidden_dim)
        target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)  # Batch_size*max_len*embedding

        context = context + target_emb_avg_exp

        tri_scores = self.feat2tri(context)  # Batch_size*sent_len*2

        # Take target embedding into consideration



        marginals = []
        select_polarities = []
        label_scores = []
        # Sentences have different lengths, so deal with them one by one
        for i, tri_score in enumerate(tri_scores):
            sent_len = lens[i].cpu().item()
            if sent_len > 1:
                tri_score = tri_score[:sent_len, :]  # sent_len, 2
            else:
                print('Too short sentence')
            marginal = self.inter_crf(tri_score)  # sent_len, latent_label_size
            # Get only the positive latent factor
            select_polarity = marginal[:, 1]  # sent_len, select only positive ones

            marginal = marginal.transpose(0, 1)  # 2 * sent_len
            sent_v = torch.mm(select_polarity.unsqueeze(0),
                              context[i, :sent_len, :])  # 1*sen_len, sen_len*hidden_dim=1*hidden_dim
            label_score = self.feat2label(sent_v).squeeze(0)  # label_size
            label_scores.append(label_score)
            select_polarities.append(select_polarity)
            marginals.append(marginal)

        label_scores = torch.stack(label_scores)

        return label_scores, select_polarities

    def compute_predict_scores(self, sents, masks, lens):
        '''
        Args:
        sents: batch_size*max_len*word_dim
        masks: batch_size*max_len
        lens: batch_size
        '''

        context = self.bilstm(sents, lens)  # Batch_size*sent_len*hidden_dim
        batch_size, max_len, hidden_dim = context.size()
        # Target embeddings
        # Find target indices, a list of indices
        target_indices, target_max_len = convert_mask_index(masks)

        # Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(hidden_dim, batch_size, max_len).transpose(0, 1).transpose(1, 2)
        target_emb = masks * context

        target_emb_avg = torch.sum(target_emb, 1) / torch.sum(masks, 1)  # Batch_size*embedding
        # Expand dimension for concatenation
        target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, hidden_dim)
        target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)  # Batch_size*max_len*embedding

        context = context + target_emb_avg_exp

        tri_scores = self.feat2tri(context)  # Batch_size*sent_len*2

        marginals = []
        select_polarities = []
        label_scores = []
        best_latent_seqs = []
        # Sentences have different lengths, so deal with them one by one
        for i, tri_score in enumerate(tri_scores):
            sent_len = lens[i].cpu().item()
            if sent_len > 1:
                tri_score = tri_score[:sent_len, :]  # sent_len, 2
            else:
                print('Too short sentence')
            marginal = self.inter_crf(tri_score)  # sent_len, latent_label_size
            best_latent_seq = self.inter_crf.predict(tri_score)  # sent_len
            # Get only the positive latent factor
            select_polarity = marginal[:, 1]  # sent_len, select only positive ones

            marginal = marginal.transpose(0, 1)  # 2 * sent_len
            sent_v = torch.mm(select_polarity.unsqueeze(0),
                              context[i][:sent_len])  # 1*sen_len, sen_len*hidden_dim=1*hidden_dim
            label_score = self.feat2label(sent_v).squeeze(0)  # label_size
            label_scores.append(label_score)
            best_latent_seqs.append(best_latent_seq)

        label_scores = torch.stack(label_scores)

        return label_scores, best_latent_seqs

    def forward(self, sents, masks, labels, lens):
        '''
        inputs are list of list for the convenince of top CRF
        Args:
        sent: a list of sentences， batch_size*len*emb_dim
        mask: a list of mask for each sentence, batch_size*len
        label: a list labels
        '''

        # scores: batch_size*label_size
        # s_prob:batch_size*sent_len
        sents = self.cat_layer(sents, masks)
        scores, s_prob = self.compute_scores(sents, masks, lens)
        s_prob_norm = torch.stack([s.norm(1) for s in s_prob]).mean()

        pena = F.relu(self.inter_crf.transitions[1, 0] - self.inter_crf.transitions[0, 0]) + \
               F.relu(self.inter_crf.transitions[0, 1] - self.inter_crf.transitions[1, 1])
        norm_pen = self.config.C1 * pena + self.config.C2 * s_prob_norm

        scores = F.log_softmax(scores, dim=1)  # Batch_size*label_size

        cls_loss = self.loss(scores, labels)

        print('Transition', pena)

        print("cls loss {0} with penalty {1}".format(cls_loss.item(), norm_pen.item()))
        return cls_loss,norm_pen

    def predict(self, sents, masks, sent_lens):
        sents = self.cat_layer(sents, masks)
        scores, best_seqs = self.compute_predict_scores(sents, masks, sent_lens)
        _, pred_label = scores.max(1)

        # Modified by Richard Sun
        return pred_label, best_seqs

    def get_target_emb(self, masks, context):
        '''
        Get the embeddings of targets
        '''
        batch_size, sent_len, dim = context.size()
        # Find target indices, a list of indices
        target_indices, target_max_len = convert_mask_index(masks)
        target_lens = [len(index) for index in target_indices]
        target_lens = torch.LongTensor(target_lens)

        # Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(dim, batch_size, sent_len).transpose(0, 1).transpose(1, 2)
        target_emb = masks * context
        # Get embeddings for each target
        if target_max_len < 3:
            target_max_len = 3
        target_embe_squeeze = torch.zeros(batch_size, target_max_len, dim)
        for i, index in enumerate(target_indices):
            target_embe_squeeze[i][:len(index)] = target_emb[i][index]
        if self.config.if_gpu:
            target_embe_squeeze = target_embe_squeeze.cuda()
            target_lens = target_lens.cuda()
        return target_embe_squeeze, target_lens


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
