import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
from CRF import LinearCRF
import torch.nn.init as init
from Layer import SimpleCat
import numpy as np
from torch.nn import utils as nn_utils
from util import *

def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

            
class biLSTM(nn.Module):
    def __init__(self, config):
        super(biLSTM, self).__init__()
        self.config = config

        self.rnn = nn.LSTM(config.embed_dim + config.mask_dim, int(config.l_hidden_size / 2), batch_first=True, num_layers = int(config.l_num_layers / 2),
            bidirectional=True)
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
        
        #batch_size*max_len*hidden_dim
        lstm_out, _ = self.rnn(pack)
        #Unpack the tensor, get the output for varied-size sentences
        #padding with zeros
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # batch * sent_l * 2 * hidden_states 
        return unpacked

# consits of three components
class ElmoAspectSent(nn.Module):
    def __init__(self, config):
        '''
        LSTM+Aspect
        '''
        super(ElmoAspectSent, self).__init__()
        self.config = config

        self.bilstm = biLSTM(config)
        self.feat2tri = nn.Linear(config.l_hidden_size, 2)
        self.inter_crf = LinearCRF(config)
        self.feat2label = nn.Linear(config.l_hidden_size, 3)

        self.loss = nn.NLLLoss()
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        #Modified by Richard Sun
        self.cat_layer = SimpleCat(config)

    
    def compute_scores(self, sents, masks, lens):
        '''
        Args:
        sents: batch_size*max_len*elmo_dim
        masks: batch_size*max_len
        lens: batch_size
        '''
        batch_size, max_len, _ = sents.size()

        context = self.bilstm(sents, lens)#batch_size*max_len*dim
        
        tri_scores = self.tanh(self.feat2tri(context)) #Batch_size*sent_len*2
        
        #Take target embedding into consideration
        
        
        
        marginals = []
        select_polarities = []
        label_scores = []
        #Sentences have different lengths, so deal with them one by one
        for i, tri_score in enumerate(tri_scores):
            sent_len = lens[i].cpu().item()
            if sent_len > 1:
                tri_score = tri_score[:sent_len, :]#sent_len, 2
            else:
                print('Too short sentence')
            marginal = self.inter_crf(tri_score)#sent_len, latent_label_size
            #Get only the positive latent factor
            select_polarity = marginal[:, 1]#sent_len, select only positive ones

            marginal = marginal.transpose(0, 1)  # 2 * sent_len
            sent_v = torch.mm(select_polarity.unsqueeze(0), context[i, :sent_len, :]) # 1*sen_len, sen_len*hidden_dim=1*hidden_dim
            label_score = self.feat2label(sent_v).squeeze(0)#label_size
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

        batch_size, max_len, _ = sents.size()
        #batch_size*target_len*emb_dim
        context = self.bilstm(sents, lens)#Batch_size*max_len*hidden_dim
        tri_scores = self.feat2tri(context) #Batch_size*max_len*2
        
        marginals = []
        select_polarities = []
        label_scores = []
        best_latent_seqs = []
        #Sentences have different lengths, so deal with them one by one
        for i, tri_score in enumerate(tri_scores):
            sent_len = lens[i].cpu().item()
            if sent_len > 1:
                tri_score = tri_score[:sent_len, :]#sent_len, 2
            else:
                print('Too short sentence')
            marginal = self.inter_crf(tri_score)#sent_len, latent_label_size
            best_latent_seq = self.inter_crf.predict(tri_score)#sent_len
            #Get only the positive latent factor
            select_polarity = marginal[:, 1]#sent_len, select only positive ones

            marginal = marginal.transpose(0, 1)  # 2 * sent_len
            sent_v = torch.mm(select_polarity.unsqueeze(0), context[i][:sent_len]) # 1*sen_len, sen_len*hidden_dim=1*hidden_dim
            label_score = self.feat2label(sent_v).squeeze(0)#label_size
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

        #scores: batch_size*label_size
        #s_prob:batch_size*sent_len
        sents = self.cat_layer(sents, masks, True)
        sents = self.dropout(sents)#regularization
        scores, s_prob  = self.compute_scores(sents, masks, lens)
        s_prob_norm = torch.stack([s.norm(1) for s in s_prob]).mean()

        pena = F.relu( self.inter_crf.transitions[1,0] - self.inter_crf.transitions[0,0]) + \
            F.relu(self.inter_crf.transitions[0,1] - self.inter_crf.transitions[1,1])
        norm_pen = self.config.C1 * pena/self.config.batch_size + self.config.C2 * s_prob_norm 

        scores = F.log_softmax(scores, dim=1)#Batch_size*label_size
        
        cls_loss = self.loss(scores, labels)

        print('Transition', pena)

        print("cls loss {0} with penalty {1}".format(cls_loss.item(), norm_pen.item()))
        return cls_loss + norm_pen 

    def predict(self, sents, masks, sent_lens):
        sents = self.cat_layer(sents, masks, True)
        scores, best_seqs = self.compute_predict_scores(sents, masks, sent_lens)
        _, pred_label = scores.max(1)    
        
        #Modified by Richard Sun
        return pred_label, scores, best_seqs
    