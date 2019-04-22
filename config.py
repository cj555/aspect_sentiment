# -*- coding: utf-8 -*-
# @Time    : 22/4/19 10:19 AM
# @Author  : CJ

import pickle
from data import DataReader, DataGenerator, DataHelper
# from data import DataReader, DataGenerator, DataHelper

class DataConfig(object):
    def __init__(self):
        self.data_path = 'data/'
        self.train = 'data/peter_train.xml'
        self.dev = 'data/peter_dev.xml'
        self.test = 'data/peter_test.xml'
        self.output_name = 'peter'
        self.embed_num = 5120  # most freq words
        self.embed_dim = 100
        self.pretrained_embed_path = 'data/glove_indo_original.100d.txt'
        self.is_stanford_nlp = False
        self.batch_size = 32
        self.pickle_path = self.data_path + self.output_name + '.pkl'


class ModelConfig(object):
    def __init__(self, data_config):
        self.epoch = 30
        self.adjust_every = 8
        self.opt = 'SGD'
        self.dropout: 0.5
        self.lr = 0.0001
        self.l2 = 0.001
        self.clip_norm = 5
        self.batch_size = data_config.batch_size
        self.embed_dim = data_config.embed_dim
        self.embed_num = data_config.embed_num
        self.l_hidden_size = 256
        self.l_num_layers = 2
        self.l_dropout= 0.1
        self.mask_dim= 20
        self.data_config = data_config
        with open(data_config.pickle_path, 'rb') as fh:
            data = pickle.load(fh)
        self.embedding = data.local_emb

    # def load_train_data(self):
    #     with open(self.data_config.pickle_path, 'rb') as fh:
    #         data = pickle.load(fh)
    #
    #     dg_train = DataGenerator(data, self.data_config, data_batch=data.data[self.data_config.train],
    #                              is_training=True)
    #     dg_dev = DataGenerator(data, self.data_config, data_batch=data.data[self.data_config.dev], is_training=True)
    #     dg_test = DataGenerator(data, self.data_config, data_batch=data.data[self.data_config.test], is_training=True)
    #
    #     return dg_train, dg_dev, dg_test


