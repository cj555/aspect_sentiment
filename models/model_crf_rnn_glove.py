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
from Layer import SimpleCat
from torch.nn import utils as nn_utils
from util import *
from data_reader_general import data_reader, data_generator


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

        # Extract the outputs for the last timestep of each example
        idx = (torch.LongTensor(seq_lengths.cpu()) - 1).view(-1, 1).expand(
            len(seq_lengths), unpacked.size(2))

        # batch first
        time_dimension = 1  # if batch_first else 0
        idx = idx.unsqueeze(time_dimension)
        if unpacked.is_cuda:
            idx = idx.cuda(unpacked.data.get_device())
        # Shape: (batch_size, rnn_hidden_dim)
        last_output = unpacked.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)

        return unpacked, last_output


class SimpleLearner(nn.Module):
    def __init__(self, config):
        super(SimpleLearner, self).__init__()
        self.feat2label = nn.Linear(config.l_hidden_size, 3)
        weight = None
        self.loss = nn.NLLLoss(weight=weight)

    def forward(self, context, labels):
        # context = self.bilstm(sents, lens)  # Batch_size*sent_len*hidden_dim
        # batch_size, max_len, hidden_dim = context.size()
        scores = self.feat2label(context)
        scores = F.log_softmax(scores, dim=1)  # Batch_size*label_size
        cls_loss = self.loss(scores, labels.cuda())
        return cls_loss


class MetaLearner(nn.Module):
    def __init__(self, config):
        '''
        LSTM+Aspect
        '''
        super(MetaLearner, self).__init__()
        self.config = config
        self.bilstm = biLSTM(config)
        self.simplelearner = SimpleLearner(config)
        self.test2target_weight = nn.Linear(config.l_hidden_size, config.aspect_size*2)
        # self.test2target_weight.weight = torch.nn.Parameter(torch.one(config.l_hidden_size, config.aspect_size))

        leaner_parameters = filter(lambda p: p.requires_grad, self.simplelearner.parameters())
        self.simpl_learner_optimizer = optim.Adam(leaner_parameters, lr=config.lr, weight_decay=config.l2)

        self.cat_layer = SimpleCat(config)
        self.cat_layer.load_vector()

    def forward(self, train_data, meta_train_target):

        domain_weight, label_list0, context0 = self.predict_domain_weight(meta_train_target,
                                                                          train_data)
        self.train_simple_leaner(domain_weight, meta_train_target, train_data)

        meta_leaner_loss = self.simplelearner(context0, label_list0)

        return meta_leaner_loss

    def train_simple_leaner(self, domain_weight, meta_train_target, train_data):
        ## training learner
        self.simplelearner.train()
        for idx, k in enumerate(self.config.aspect_name_ix.keys()):
            if k == meta_train_target:
                continue
            dg_learner_train = data_generator(self.config, train_data[k])
            sent_vecs1, mask_vecs1, label_list1, sent_lens1, _, target_name_list1, _ = next(
                dg_learner_train.get_ids_samples())
            label_list1 = torch.LongTensor(label_list1)
            weight = torch.mean(domain_weight[:, idx])
            # weight = domain_weight[idx]
            sent_vecs1 = self.cat_layer(sent_vecs1, mask_vecs1)
            _, context1 = self.bilstm(sent_vecs1, sent_lens1)  # Batch_size*hidden_dim, last hidden states
            batch_size, hidden_dim = context1.size()

            print(meta_train_target, k, weight)
            leaner_loss = self.simplelearner(context1, label_list1)
            leaner_loss *= (0.5+weight)
            self.simplelearner.zero_grad()
            leaner_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.simplelearner.parameters(), self.config.clip_norm, norm_type=2)
            self.simpl_learner_optimizer.step()

    def predict_domain_weight(self, meta_train_target, train_data):
        dg_meta_train = data_generator(self.config, train_data[meta_train_target])
        sent_vecs0, mask_vecs0, label_list0, sent_lens0, _, target_name_list0, _ = next(
            dg_meta_train.get_ids_samples())
        label_list0 = torch.LongTensor(label_list0)
        sent_vecs0 = self.cat_layer(sent_vecs0, mask_vecs0)
        _, context0 = self.bilstm(sent_vecs0, sent_lens0)  # Batch_size*hidden_dim, last hidden states
        batch_size, hidden_dim = context0.size()
        meta_scores = self.test2target_weight(context0)
        domain_weight = F.softmax(meta_scores.view(meta_scores.shape[0], meta_scores.shape[1] // 2, 2), dim=2)[:, :, 0]
        # domain_weight = F.softmax(meta_scores,dim=1)*self.config.aspect_size
        # domain_weight = F.softmax(torch.sum(meta_scores,dim=0))*self.config.aspect_size

        return domain_weight, label_list0, context0


# consits of three components
class CRFAspectSent(nn.Module):
    def __init__(self, config, bilstm):
        '''
        LSTM+Aspect
        '''
        super(CRFAspectSent, self).__init__()
        self.config = config
        # self.bilstm = biLSTM(config)
        self.bilstm = bilstm
        self.feat2tri = nn.Linear(config.l_hidden_size, 2)
        self.inter_crf = LinearCRF(config)
        self.feat2label = nn.Linear(config.l_hidden_size, 3)
        # self.feat2target = nn.Linear(config.l_hidden_size, len(config.train_targets))
        # self.feat2target = nn.Linear(config.l_hidden_size, 3)
        # self.target_loss = nn.NLLLoss()
        # self.target_embed = nn.Embedding( len(config.train_targets), config.mask_dim)


        # self.feat2sourcetarget = nn.Linear(config.l_hidden_size, 2)
        # self.sourcetarget_loss = nn.NLLLoss()
        # self.sourcetarget_loss = nn.MSELoss()

        # weight = torch.Tensor([0.16, 0.71, 0.13])

        # weight = 1 / weight
        weight = None
        self.loss = nn.NLLLoss(weight=weight)

        self.cat_layer = SimpleCat(config)
        self.cat_layer.load_vector()
        # Modified by Richard Sun

    def compute_scores(self, sents, masks, lens, mode='sentiment_classifier'):
        '''
        Args:
        sents: batch_size*max_len*word_dim
        masks: batch_size*max_len
        lens: batch_size
        '''

        # Context embeddings
        context,_ = self.bilstm(sents, lens)  # Batch_size*sent_len*hidden_dim

        # target_scores = self.feat2target(context[:,-1,:])
        # target_log_probs = F.log_softmax(target_scores,dim=1)


        batch_size, max_len, hidden_dim = context.size()
        # Target embeddings
        # Find target indices, a list of indices
        target_indices, target_max_len = convert_mask_index(masks)

        # Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(hidden_dim, batch_size, max_len).transpose(0, 1).transpose(1, 2)
        masks = -(masks - 1) * torch.ones(masks.shape).type_as(
            masks)  # reverse target embedding since targets are missing in test, we use context around target to represent target

        target_emb = masks * context

        true_target_emb_avg = torch.sum(target_emb, 1) / torch.sum(masks, 1)  # Batch_size*embedding

        if mode == 'target_cls':
            # target_scores = self.feat2sourcetarget(true_target_emb_avg)
            # target_log_probs = F.log_softmax(target_scores, dim=1)
            # return None, None, target_log_probs

            # supervised on target classification
            target_scores = self.feat2target(true_target_emb_avg)
            target_log_probs = F.log_softmax(target_scores, dim=1)
            # cls_loss = self.target_loss(target_log_probs)
            # supervised on target regression
            # target_log_probs = target_scores

            return target_log_probs, None, true_target_emb_avg
        elif mode == 'sentiment_classifier':

            # Expand dimension for concatenation
            true_target_emb_avg_exp = true_target_emb_avg.expand(max_len, batch_size, hidden_dim)
            true_target_emb_avg_exp = true_target_emb_avg_exp.transpose(0, 1)  # Batch_size*max_len*embedding

            context = context + true_target_emb_avg_exp

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

            return label_scores, select_polarities, None

    def compute_predict_scores(self, sents, masks, lens):
        '''
        Args:
        sents: batch_size*max_len*word_dim
        masks: batch_size*max_len
        lens: batch_size
        '''

        context,_ = self.bilstm(sents, lens)  # Batch_size*sent_len*hidden_dim
        batch_size, max_len, hidden_dim = context.size()
        # Target embeddings
        # Find target indices, a list of indices
        target_indices, target_max_len = convert_mask_index(masks)

        # Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(hidden_dim, batch_size, max_len).transpose(0, 1).transpose(1, 2)
        masks = -(masks - 1) * torch.ones(masks.shape).type_as(masks)

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

        scores, s_prob, target_embeding = self.compute_scores(sents, masks, lens, mode='sentiment_classifier')

        s_prob_norm = torch.stack([s.norm(1) for s in s_prob]).mean()

        pena = F.relu(self.inter_crf.transitions[1, 0] - self.inter_crf.transitions[0, 0]) + \
               F.relu(self.inter_crf.transitions[0, 1] - self.inter_crf.transitions[1, 1])
        norm_pen = self.config.C1 * pena + self.config.C2 * s_prob_norm

        scores = F.log_softmax(scores, dim=1)  # Batch_size*label_size

        cls_loss = self.loss(scores, labels)

        cls_loss = cls_loss + 1e-2

        print('Transition', pena)

        print("cls loss {0} with penalty {1}".format(cls_loss.item(), norm_pen.item()))

        return cls_loss + norm_pen

    def predict(self, sents, masks, sent_lens):
        sents = self.cat_layer(sents, masks)
        scores, best_seqs = self.compute_predict_scores(sents, masks, sent_lens)
        _, pred_label = scores.max(1)

        # Modified by Richard Sun
        return pred_label, best_seqs

    def get_target_emb(self, context, masks):
        # Target embeddings
        # Find target indices, a list of indices
        batch_size, max_len, hidden_dim = context.size()
        target_indices, target_max_len = convert_mask_index(masks)

        # Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(hidden_dim, batch_size, max_len).transpose(0, 1).transpose(1, 2)
        target_emb = masks * context
        target_emb_avg = torch.sum(target_emb, 1) / torch.sum(masks, 1)  # Batch_size*embedding
        return target_emb_avg


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
