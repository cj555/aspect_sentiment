import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
import pickle
from BatchLinearChainCRF import LinearChainCrf
import torch.nn.init as init
import numpy as np
from torch.nn import utils as nn_utils
from util import *
from Layer import SimpleCat, SimpleCatTgtMasked
from multiprocessing import Pool
import copy


# torch.manual_seed(222)
def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)


class biLSTM(nn.Module):
    def __init__(self, config):
        super(biLSTM, self).__init__()
        self.config = config

        self.rnn = nn.GRU(config.embed_dim + config.mask_dim, int(config.l_hidden_size / 2), batch_first=True,
                          num_layers=int(config.l_num_layers / 2),
                          bidirectional=True)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        """

        :param feats:batch_size, max_len, emb_dim
        :param seq_lengths:batch_size
        :return:
        """
        pack = nn_utils.rnn.pack_padded_sequence(feats,
                                                 seq_lengths, batch_first=True)

        # batch_size*max_len*hidden_dim
        lstm_out, _ = self.rnn(pack)
        # Unpack the tensor, get the output for varied-size sentences
        # padding with zeros
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # batch * sent_l * 2 * hidden_states 
        return unpacked


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


# consits of three components
class AspectSent(nn.Module):
    def __init__(self, config, mode='sent_cls'):
        """
         LSTM+Aspect
        :param config:
        """

        super(AspectSent, self).__init__()
        self.config = config

        input_dim = config.l_hidden_size
        kernel_num = config.l_hidden_size
        # self.conv = nn.Conv1d(input_dim, kernel_num, 3, padding=1)
        # self.conv = nn.Conv1d(input_dim, kernel_num, 3, dilation=2, padding=2)

        self.bilstm = biLSTM(config)

        ## adc:Adversarial Domain Classification
        self.feat2tri_adc = nn.Linear(kernel_num, 2 + 2)
        self.inter_crf_adc = LinearChainCrf(2 + 2)

        self.feat2sent_adc = nn.Linear(kernel_num, 3)
        self.feat2domain_adc = nn.Linear(kernel_num, 2)

        ## dc:Domain Classification
        self.feat2tri_dc = nn.Linear(kernel_num, 2 + 2)
        self.inter_crf_dc = LinearChainCrf(2 + 2)

        self.feat2sent_dc = nn.Linear(kernel_num, 3)
        self.feat2domain_dc = nn.Linear(kernel_num, 2)
        self.feat2sent = nn.Linear(kernel_num * 2, 3)

        self.loss = nn.NLLLoss()

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout)

        # Modified by Richard Sun
        self.cat_layer = SimpleCat(config)
        self.cat_layer.load_vector()

    def get_pos_weight(self, masks, lens):
        """
        Get positional weight
        :param masks:
        :param lens:
        :return:
        """

        pos_wghts = torch.zeros(masks.size())
        t_num = masks.sum(1)
        for i, m in enumerate(masks):
            begin = m.argmax()
            for j, b in enumerate(m):
                # padding words' weights are zero
                if j > lens[i]:
                    break
                if j < begin:
                    pos_wghts[i][j] = 1 - (begin - j).to(torch.float) / lens[i].to(torch.float)
                if b == 1:
                    pos_wghts[i][j] = 1
                if j > begin + t_num[i]:
                    pos_wghts[i][j] = 1 - (j - begin).to(torch.float) / lens[i].to(torch.float)
        return pos_wghts

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

    def compute_scores(self, sents, masks, lens, mode, is_training=True):
        """

        :param sents:batch_size*max_len*word_dim
        :param masks:batch_size*max_len
        :param lens:batch_size
        :param is_training:
        :return:
        """
        context = self.bilstm(sents, lens)  # Batch_size*sent_len*hidden_dim
        # pos_weights = self.get_pos_weight(masks, lens)#Batch_size*sent_len
        # context = torch.cat([context, sents[:, :, :-30]], 2)#Batch_size*sent_len*(hidden_dim+word_embed)

        #         context = F.relu(self.conv(context.transpose(1, 2)))
        #         context = context.transpose(1, 2)
        # context = torch.tanh(context)

        # att1_weight = F.softmax(self.attn1(context), dim=1)
        # domain_specific_context = att1_weight.expand(-1, -1,
        #                                              self.config.l_hidden_size) * context  # batch X seq_len X hidden_dim
        # inverse_att1_weight = F.softmax(1 / (att1_weight + 1e-5), 1)
        # domain_share_context = inverse_att1_weight.expand(-1, -1, self.config.l_hidden_size) * context
        #
        batch_size, max_len, hidden_dim = context.size()

        # Expand dimension for concatenation
        target_emb_avg = self.get_target_emb(torch.tanh(context), masks)
        target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, hidden_dim)
        target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)  # Batch_size*max_len*embedding

        best_latent_seqs, label_scores, select_polarities, context = self._compute_crf_scores(batch_size,
                                                                                              torch.tanh(context),
                                                                                              lens,
                                                                                              max_len,
                                                                                              target_emb_avg_exp,
                                                                                              mode=mode)

        # print('{},best_latent_seq\n{}'.format(mode, best_latent_seqs[:3]))

        # best_latent_seqs_share, domain_share_label_scores, _ = self._compute_crf_scores(batch_size, torch.tanh(
        #     domain_share_context), lens,
        #                                                                                 max_len, target_emb_avg_exp)
        # best_latent_seqs_specific, domain_specific_label_scores, _ = self._compute_crf_scores(batch_size, torch.tanh(
        #     domain_specific_context), lens,
        #                                                                                       max_len,
        #                                                                                       target_emb_avg_exp)
        #
        # domain_disagreement = domain_share_label_scores - domain_specific_label_scores

        # if mode == 'domain cls':
        #     test_seqs = [(best_latent_seqs[idx], best_latent_seqs_share[idx], best_latent_seqs_specific[idx]) for idx, i
        #                  in enumerate(labels == 2) if i == 1]
        #
        #     print('test,{},best_latent_seq:full,share,specific\n{}\n{}\n{}'.format(mode, test_seqs[0][0],
        #                                                                            test_seqs[0][1],
        #                                                                            test_seqs[0][2]))
        #
        #     train_seqs = [(best_latent_seqs[idx], best_latent_seqs_share[idx], best_latent_seqs_specific[idx]) for
        #                   idx, i
        #                   in enumerate(labels == 0) if i == 1]
        #
        #     print('train,{},best_latent_seq:full,share,specific\n{}\n{}\n{}'.format(mode, train_seqs[0][0],
        #                                                                             train_seqs[0][1],
        #                                                                             train_seqs[0][2]))
        # elif mode == 'sent_cls':
        #     print('train,{},best_latent_seq:full,share,specific\n{}\n{}\n{}'.format(mode, best_latent_seqs[0],
        #                                                                             best_latent_seqs_share[0],
        #                                                                             best_latent_seqs_specific[
        #                                                                                 0]))

        if is_training:
            return label_scores, select_polarities, context
        else:
            return label_scores, best_latent_seqs, context

    def _compute_crf_scores(self, batch_size, context, lens, max_len, target_emb_avg_exp, mode='adc'):
        if mode == 'adc' or mode == 'sent_cls_adc':
            ###Addition model
            u = target_emb_avg_exp
            context = context + u  # Batch_size*max_len*embedding

            word_mask = torch.full((batch_size, max_len), 0)
            for i in range(batch_size):
                word_mask[i, :lens[i]] = 1.0
            ###neural features
            feats = self.feat2tri_adc(context)  # Batch_size*sent_len*2
            marginals = self.inter_crf_adc.compute_marginal(feats, word_mask.type_as(feats))
            # print(word_mask.sum(1))
            select_polarities = [marginal[:, 1] for marginal in marginals]
            gammas = [sp.sum() / 2 for sp in select_polarities]
            sent_vs = [torch.mm(sp.unsqueeze(0), context[i, :lens[i], :]) for i, sp in enumerate(select_polarities)]
            sent_vs = [sv / gamma for sv, gamma in zip(sent_vs, gammas)]  # normalization
            sent_vs = torch.cat(sent_vs)  # batch_size, hidden_size

            sent_vs = self.dropout(sent_vs)

            label_scores = self.feat2sent_adc(sent_vs) if mode == 'sent_cls_adc' else self.feat2domain_adc(sent_vs)

            best_latent_seqs = self.inter_crf_adc.decode(feats, word_mask.type_as(feats))
            # print(best_latent_seqs[0])
            return best_latent_seqs, label_scores, select_polarities, sent_vs

        elif mode == 'dc' or mode == 'sent_cls_dc':
            ###Addition model
            u = target_emb_avg_exp
            context = context + u  # Batch_size*max_len*embedding

            word_mask = torch.full((batch_size, max_len), 0)
            for i in range(batch_size):
                word_mask[i, :lens[i]] = 1.0
            ###neural features
            feats = self.feat2tri_dc(context)  # Batch_size*sent_len*2
            marginals = self.inter_crf_dc.compute_marginal(feats, word_mask.type_as(feats))
            # print(word_mask.sum(1))
            select_polarities = [marginal[:, 1] for marginal in marginals]
            gammas = [sp.sum() / 2 for sp in select_polarities]
            sent_vs = [torch.mm(sp.unsqueeze(0), context[i, :lens[i], :]) for i, sp in enumerate(select_polarities)]
            sent_vs = [sv / gamma for sv, gamma in zip(sent_vs, gammas)]  # normalization
            sent_vs = torch.cat(sent_vs)  # batch_size, hidden_size

            sent_vs = self.dropout(sent_vs)

            label_scores = self.feat2sent_dc(sent_vs) if mode == 'sent_cls_dc' else self.feat2domain_dc(sent_vs)

            best_latent_seqs = self.inter_crf_dc.decode(feats, word_mask.type_as(feats))
            # print(best_latent_seqs[0])
            return best_latent_seqs, label_scores, select_polarities, sent_vs
        else:
            raise ValueError('no such mode')

    def forward(self, sents, masks, labels, lens, mode='sent_cls', lambd=1.0, dc_context=None, adc_context=None):
        """
        inputs are list of list for the convenince of top CRF
        :param sents: a list of sentences， batch_size*len*emb_dim
        :param masks: a list of mask for each sentence, batch_size*len
        :param labels: a list sentiment labels or a list of source labels
        :param lens: 
        :return: 
        """

        # scores: batch_size*label_size
        # s_prob:batch_size*sent_len

        # sents = grad_reverse(sents)
        # masks = grad_reverse(masks)
        # labels = grad_reverse(labels)

        if self.config.if_reset:  self.cat_layer.reset_binary()
        sents = self.cat_layer(sents, masks)

        if mode == 'sent_cls_adc' or mode == 'adc' or mode == 'dc' or mode == 'sent_cls_dc':
            scores, s_prob, context = self.compute_scores(sents, masks, lens, mode=mode)
            s_prob_norm = torch.stack([s.norm(1) for s in s_prob]).mean()

            pena = F.relu(self.inter_crf_adc.transitions[1, 0] - self.inter_crf_adc.transitions[0, 0]) + \
                   F.relu(self.inter_crf_adc.transitions[0, 1] - self.inter_crf_adc.transitions[1, 1])
            norm_pen = self.config.C1 * pena + self.config.C2 * s_prob_norm

            # print('Transition Penalty:', pena)
            # print('Marginal Penalty:', s_prob_norm)
            if mode == 'adc':
                scores = grad_reverse(scores, lambd)
            scores = F.log_softmax(scores, 1)  # Batch_size*label_size

            cls_loss = self.loss(scores, labels)

            return cls_loss, norm_pen, scores, context
        elif mode == 'sent_cls':
            context = torch.cat([dc_context, adc_context], dim=1)
            scores = self.feat2sent(context)
            scores = F.log_softmax(scores, 1)  # Batch_size*label_size

            cls_loss = self.loss(scores, labels)
            return cls_loss, None, None, None

    # elif mode == 'domain_cls':
    #     # if self.config.if_reset:  self.cat_layer.reset_binary()
    #     # sents = self.cat_layer(sents, masks)
    #     context = self.bilstm(sents, lens)  # batch X sentence_len X hidden_dim
    #     # context = torch.tanh(context)
    #     att1_weight = F.softmax(self.attn1(context), dim=1)
    #     domain_specific_context = torch.mean(att1_weight.expand(-1, -1, self.config.l_hidden_size) * context,
    #                                          dim=1)  # batch X hidden_dim
    #     domain_specific_context = torch.tanh(domain_specific_context)
    #     domain_cls_score = self.domaincls(domain_specific_context)
    #
    #     # if domain_adapt_mode == 'cls':
    #
    #     ## domain classification
    #
    #     domain_cls_loss = self.loss(F.log_softmax(domain_cls_score, 1), labels)
    #     unsuper_loss = torch.zeros(1).cuda(device=self.config.gpu)
    #     return domain_cls_loss, unsuper_loss
    # elif mode == 'da':
    #     context = self.bilstm(sents, lens)  # batch X sentence_len X hidden_dim
    #     context = torch.tanh(context)
    #     att1_weight = F.softmax(self.attn1(context), dim=1)
    #     domain_specific_context = torch.mean(att1_weight.expand(-1, -1, self.config.l_hidden_size) * context,
    #                                          dim=1)  # batch X hidden_dim
    #
    #     domain_cls_score = self.domaincls(domain_specific_context)
    #     domain_cls_loss = self.loss(F.log_softmax(domain_cls_score, 1), labels)
    #
    #     labels_score, labels_idx = torch.max(F.softmax(domain_cls_score, 1), 1)
    #
    #     ## unsupervised
    #     threshhold = self.config.sentagree_thresh
    #     # inverse_att1_weight = F.softmax(1 / (att1_weight + 1e-5), 1)
    #     # domain_share_context = torch.mean(
    #     #     inverse_att1_weight.expand(-1, -1, self.config.l_hidden_size) * context,
    #     #     dim=1)  # batch X hidden_dim
    #
    #     # domain_specific_view = domain_specific_context[(labels_score > threshhold) & (labels_idx == labels)]
    #     # domain_share_view = domain_share_context[(labels_score > threshhold) & (labels_idx == labels)]
    #     unsuper_loss = torch.zeros(1).cuda(device=self.config.gpu)
    #     if torch.sum((labels_score > threshhold) & (labels_idx == labels)) > 0:
    #         _, _, domain_disagreement = self.compute_scores(sents, masks, lens, mode=mode, labels=labels)
    #         domain_disagreement = domain_disagreement[(labels_score > threshhold) & (labels_idx == labels)]
    #         unsuper_loss = torch.mean(torch.norm(domain_disagreement, dim=1, p=2))
    #
    #     return domain_cls_loss, unsuper_loss

    # elif domain_adapt_mode == 'unsupervised':
    #
    #     domain_share_context = torch.mean(inverse_att1_weight.expand(-1, -1, self.config.l_hidden_size) * context,
    #                                          dim=1)  # batch X hidden_dim
    #
    #     domain_specific_context[(labels_score > 0.5) & (labels_idx == labels)]

    # train_labels = torch.zeros(labels.shape).type(labels.type())
    # train_labels[labels == 0] = 1
    #
    # valid_labels = torch.zeros(labels.shape).type(labels.type())
    # valid_labels[labels == 1] = 1
    #
    # test_labels = torch.zeros(labels.shape).type(labels.type())
    # test_labels[labels == 2] = 1
    #
    # train_context = self.is_train(domain_context)
    # valid_context = self.is_valid(domain_context)
    # test_context = self.is_test(domain_context)
    #
    # train_cls_loss = self.loss(F.log_softmax(train_context, 1), train_labels)
    # valid_cls_loss = self.loss(F.log_softmax(valid_context, 1), valid_labels)
    # test_cls_loss = self.loss(F.log_softmax(test_context, 1), test_labels)
    #
    # domain_cls_loss = (train_cls_loss + valid_cls_loss + test_cls_loss) / 3
    #

    # elif domain_adapt_mode== 'unsupervised':
    #     self.predict_domain(sents,masks,labels,lens)
    #     return domain_cls_loss

    # def predict_domain(self, sents, masks, labels, lens):
    # sents = self.cat_layer(sents, masks)
    # context = self.bilstm(sents, lens)  # batch X sentence_len X hidden_dim
    # att1_weight = F.softmax(self.attn1(context), dim=1)
    # domain_context = torch.mean(att1_weight.expand(-1, -1, self.config.l_hidden_size) * context,
    #                             dim=1)  # batch X hidden_dim
    #
    # train_labels = torch.zeros(labels.shape).type(labels.type())
    # train_labels[labels == 0] = 1
    #
    #
    # valid_labels = torch.zeros(labels.shape).type(labels.type())
    # valid_labels[labels == 1] = 1
    #
    # test_labels = torch.zeros(labels.shape).type(labels.type())
    # test_labels[labels == 2] = 1
    #
    # train_context = self.is_train(domain_context)
    # valid_context = self.is_valid(domain_context)
    # test_context = self.is_test(domain_context)
    #
    # train_labels_score,train_labels_idx  = torch.max(F.softmax(train_context, 1), 1)
    # train_labels_score,valid_labels_idx = torch.max(F.softmax(valid_context, 1), 1)
    # train_labels_score,valid_labels_idx = torch.max(F.softmax(test_context, 1), 1)
    #
    # scores = torch.cat((train_labels_score, train_labels_score, train_labels_score)).view(32, -1)

    # return

    def predict(self, sents, masks, sent_lens):
        if self.config.if_reset:  self.cat_layer.reset_binary()
        sents = self.cat_layer(sents, masks)
        scores_adc, best_seqs_adc, adc_context = self.compute_scores(sents, masks, sent_lens, mode='sent_cls_adc',
                                                                     is_training=False)
        scores_dc, best_seqs_dc, dc_context = self.compute_scores(sents, masks, sent_lens, mode='sent_cls_dc',
                                                                  is_training=False)
        context = torch.cat([dc_context, adc_context], dim=1)
        scores = self.feat2sent(context)
        scores = F.log_softmax(scores, 1)  # Batch_size*label_siz
        _, pred_label = scores.max(1)

        # Modified by Richard Sun
        return pred_label, best_seqs_adc, best_seqs_dc


def convert_mask_index(masks):
    """
     Find the indice of none zeros values in masks, namely the target indice
    :param masks:
    :return:
    """

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
