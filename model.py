import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils as nn_utils
from util import *
from torch.nn import utils as nn_utils
import pickle
from torch.autograd import Variable

import torch.nn.init as init


def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)


class MLSTM(nn.Module):
    def __init__(self, config):
        super(MLSTM, self).__init__()
        self.config = config
        # The concatenated word embedding and target embedding as input
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
        # FIXIT: doesn't have batch
        # Sort the lengths
        pack = nn_utils.rnn.pack_padded_sequence(feats,
                                                 seq_lengths, batch_first=True)

        # assert self.batch_size == batch_size
        lstm_out, _ = self.rnn(pack)
        # lstm_out, (hid_states, cell_states) = self.rnn(feats)

        # Unpack the tensor, get the output for varied-size sentences
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return unpacked


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


class Tri_RNNCNNSent(nn.Module):
    def __init__(self, config):
        '''
        In this model, only context words are processed by gated CNN, target is average word embeddings
        '''
        super(Tri_RNNCNNSent, self).__init__()
        self.config = config

        # V = config.embed_num
        D = config.l_hidden_size
        C = 3  # config.class_num

        Co = 128  # kernel numbers
        Ks = [2, 3, 4]  # kernel filter size
        Kt = [2, 3]  # kernel filter size for target words

        self.lstm = MLSTM(config)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K - 2) for K in Kt])

        self.convs4 = nn.ModuleList([nn.Conv1d(Co, Co, K) for K in Ks])
        self.convs5 = nn.ModuleList([nn.Conv1d(Co, Co, K) for K in Ks])

        self.fc_aspect = nn.Linear(len(Kt) * Co, Co)

        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc2 = nn.Linear(len(Ks) * Co, C)
        self.fc3 = nn.Linear(len(Ks) * Co, C)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.cat_layer = SimpleCat(config)
        self.cat_layer.load_vector()  # init word embeding
        # self.lambda_unsupervised = nn.Parameter(torch.tensor([0.1]).cuda(),requires_grad = True)
        # self.lambda_balance= nn.Parameter(torch.tensor([0.0]).cuda(),requires_grad = True)
        # self.lambda_f = nn.Parameter(torch.tensor([1.0]).cuda(), requires_grad=True)

    def compute_score(self, sents, masks, lens):
        '''
        inputs are list of list for the convenince of top CRF
        Args:
        sents: a list of sentences， batch_size*max_len*(2*emb_dim)
        masks: a list of binary to indicate target position, batch_size*max_len
        label: a list labels
        '''
        # Get the rnn outputs for each word, batch_size*max_len*hidden_size
        context = self.lstm(sents, lens)

        # Get the target embedding
        batch_size, sent_len, dim = context.size()

        # Find target indices, a list of indices
        target_indices, target_max_len = convert_mask_index(masks)

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
        if self.config.if_gpu: target_embe_squeeze = target_embe_squeeze.cuda()

        # Conv input: batch_size * emb_dim * max_len
        # Conv output: batch_size * out_dim * (max_len-k+1)
        target_conv = [self.relu(conv(target_embe_squeeze.transpose(1, 2))) for conv in
                       self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in target_conv]  # [(batch_size,Co), ...]*len(Ks)
        aspect_v = torch.cat(aa, 1)  # N, Co*len(K)
        aspect_v = self.fc_aspect(aspect_v)

        #         x = [F.tanh(conv(context.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        #         #y = [F.relu(conv(context.transpose(1, 2)) + aspect_v.unsqueeze(2)) for conv in self.convs2]
        #         y = [F.relu(aspect_v.unsqueeze(2)) for conv in self.convs2]
        #         x = [i*j for i, j in zip(x, y)] #batch_size, Co, len-1 .  batch_size, Co, len-2

        x = [conv(context.transpose(1, 2)) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.tanh(conv(context.transpose(1, 2)) + aspect_v.unsqueeze(2)) for conv in self.convs2]
        # y = [F.sigmoid(aspect_v.unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]  # batch_size, Co, len-1 .  batch_size, Co, len-2

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,out_dim), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]  # batch_size*2Co

        sents_vec = torch.cat(x0, 1)  # N*(3*out_dim)
        # logit = self.fc1(x0)  # (N,C)

        # Dropout
        if self.training:
            sents_vec = F.dropout(sents_vec, self.config.dropout)

        output1 = self.fc1(sents_vec)  # Bach_size*label_size
        output2 = self.fc2(sents_vec)
        output3 = self.fc3(sents_vec)
        # output1 = F.log_softmax(output1, dim=1)#Batch_size*label_size
        # output2 = F.log_softmax(output2, dim=1)
        f_loss = torch.mean(self.fc1.weight * self.fc2.weight)
        return output1, output2, output3, f_loss

    def forward(self, sents, masks, labels, lens):
        # Sent emb_dim
        sents = self.cat_layer(sents, masks)  # mask embedding
        # sents, _ = self.cat_layer(sents, masks)
        # sents = F.dropout(sents, p=0.5, training=self.training)
        output1, output2, output3, f_loss = self.compute_score(sents, masks, lens)

        # print('Transition', pena)
        return output1, output2, output3, f_loss

    def predict(self, sents, masks, sent_lens):
        sents = self.cat_layer(sents, masks)  # mask embedding
        # sents, _ = self.cat_layer(sents, masks)
        output1, output2, output3, f_loss = self.compute_score(sents, masks, sent_lens)

        pred_label3 = torch.max(torch.nn.functional.softmax(output3), dim=1)[1]
        # Modified by Richard Sun
        pred_label1 = torch.max(torch.nn.functional.softmax(output1), dim=1)[1]
        return output1, output2, output3, pred_label3,pred_label1


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
