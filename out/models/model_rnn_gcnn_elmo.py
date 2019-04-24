import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils as nn_utils
from util import *
from torch.nn import utils as nn_utils
import torch.nn.init as init
def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

class MLSTM(nn.Module):
    def __init__(self, config):
        super(MLSTM, self).__init__()
        self.config = config
        #The concatenated word embedding and target embedding as input
        self.rnn = nn.LSTM(config.embed_dim , int(config.l_hidden_size / 2), batch_first=True, num_layers = int(config.l_num_layers / 2),
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
        pack = nn_utils.rnn.pack_padded_sequence(feats, 
                                                 seq_lengths, batch_first=True)
        
        
        #assert self.batch_size == batch_size
        lstm_out, _ = self.rnn(pack)
        #lstm_out, (hid_states, cell_states) = self.rnn(feats)

        #Unpack the tensor, get the output for varied-size sentences
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return unpacked


class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, config):
        '''
        In this model, only context words are processed by gated CNN, target is average word embeddings
        '''
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.config = config
        
        #V = config.embed_num
        D = config.embed_dim
        C = 3#config.class_num

        Co = 128#kernel numbers
        Ks = [2, 3, 4]#kernel filter size

        self.lstm = MLSTM(config)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K-2) for K in [3]])

        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    
    def compute_score(self, sents, masks, lens):
        '''
        inputs are list of list for the convenince of top CRF
        Args:
        sents: a list of sentences， batch_size*max_len*(2*emb_dim)
        masks: a list of binry to indicate target position, batch_size*max_len
        label: a list labels
        '''
        #Get the rnn outputs for each word, batch_size*max_len*hidden_size
        context = sents

        #Get the target embedding
        batch_size, sent_len, dim = context.size()
        
        
        
        #Find target indices, a list of indices
        target_indices, target_max_len = convert_mask_index(masks)

        #Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(dim, batch_size, sent_len).transpose(0, 1).transpose(1, 2)
        target_emb = masks * context
        #Get embeddings for each target
        target_embe_squeeze = torch.zeros(batch_size, target_max_len, dim)
        for i, index in enumerate(target_indices):
            target_embe_squeeze[i][:len(index)] = target_emb[i][index]
        if self.config.if_gpu: target_embe_squeeze = target_embe_squeeze.cuda()
        
        #Conv input: batch_size * emb_dim * max_len
        #Conv output: batch_size * out_dim * (max_len-k+1)
        target_conv = [self.relu(conv(target_embe_squeeze.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in target_conv]# [(batch_size,Co), ...]*len(Ks)
        aspect_v = torch.cat(aa, 1)#N, Co*len(K)
        
        x = [F.tanh(conv(context.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(context.transpose(1, 2)) + aspect_v.unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)] #batch_size * out_dim * (max_len-k+1) * len(filters)

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,out_dim), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        sents_vec = torch.cat(x0, 1)#N*(3*out_dim)
        #logit = self.fc1(x0)  # (N,C)

        #Dropout
        if self.training:
            sents_vec = F.dropout(sents_vec, self.config.dropout)

        output = self.fc1(sents_vec)#Bach_size*label_size

        scores = F.log_softmax(output, dim=1)#Batch_size*label_size
        return scores


    def forward(self, sents, masks, labels, lens):
        #Sent emb_dim + 50
        ##For restaurant
        #sent = F.dropout(sents, p=0.2, training=self.training)
        
        #For laptop
        sent = F.dropout(sents, p=0.5, training=self.training)
        
        scores = self.compute_score(sents, masks, lens)
        loss = nn.NLLLoss()
        #cls_loss = -1 * torch.log(scores[label])
        cls_loss = loss(scores, labels)

        #print('Transition', pena)
        return cls_loss 

    def predict(self, sents, masks, sent_lens):
        #sent = self.cat_layer(sent, mask)
        scores = self.compute_score(sents, masks, sent_lens)
        _, pred_labels = scores.max(1)#Find the max label in the 2nd dimension
        
        #Modified by Richard Sun
        return pred_labels

    
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
        print(mask)
    return target_indice, max_len