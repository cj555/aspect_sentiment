#############################################
# This file aims to preprocess data and generate samples for training and testing
from collections import namedtuple, defaultdict
import codecs
# from config import config
from bs4 import BeautifulSoup
import pdb
import torch
import numpy as np
import re
import pickle
import random
import os
from collections import Counter
##Added by Richard Sun
from allennlp.modules.elmo import Elmo, batch_to_ids
import en_core_web_sm
# from config import DataConfig

nlp = en_core_web_sm.load()

SentInst = namedtuple("SentenceInstance", "id text text_ids text_inds opinions")
OpinionInst = namedtuple("OpinionInstance", "target_text polarity class_ind target_mask target_ids target_tokens")


class DataHelper():
    def __init__(self, config):
        '''
        This class is able to:
        1. Load datasets
        2. Split sentences into words
        3. Map words into Idx
        '''
        self.config = config

        # id map to instance
        try:
            self.id2label = config.labels
        except:
            self.id2label = ["positive", "neutral", "negative"]

        self.label2id = {v: k for k, v in enumerate(self.id2label)}

        self.UNK = "unk"
        self.EOS = "<eos>"
        self.PAD = "<pad>"

        # data
        self.train_data = None
        self.test_data = None

        # if config.is_stanford_nlp:
        #     from stanfordcorenlp import StanfordCoreNLP
        #     self.stanford_nlp = StanfordCoreNLP(r'../data/stanford-corenlp-full-2018-02-27')

    def read_csv_data(self, file_name):
        '''
        Read CSV data
        Args:
        file_name: path of the csv file
        text: text column name
        target: target column name
        label: label column name
        '''
        import pandas as pd
        data = pd.read_csv(file_name)
        data_num = data.shape[0]
        print('CSV Data Num:', data_num)
        sentence_list = []
        for i in np.arange(data_num):
            sent_id = i
            sent_text = self.clean_text(data['text'][i])
            opinion_list = []
            term = self.clean_text(data['target'][i])
            if pd.isnull(data['label'][i]):
                print('the ' + str(i) + 'data label is empty!!!!')
                continue
            polarity = data['label'][i].lower()
            opinion_inst = OpinionInst(term, polarity, None, None, None, None)
            opinion_list.append(opinion_inst)
            sent_Inst = SentInst(sent_id, sent_text, None, None, opinion_list)
            sentence_list.append(sent_Inst)
        return sentence_list

    def read_xml_data(self, file_name):
        '''
        Read XML data
        '''
        f = codecs.open(file_name, "r", encoding="utf-8")
        soup = BeautifulSoup(f.read(), "lxml")
        sentence_tags = soup.find_all("sentence")
        sentence_list = []
        for sent_tag in sentence_tags:
            sent_id = sent_tag.attrs["id"]
            sent_text = self.clean_text(sent_tag.find("text").contents[0])
            opinion_list = []
            try:
                asp_tag = sent_tag.find_all("aspectterms")[0]
            except:
                # print "{0} {1} has no opinions".format(sent_id, sent_text)
                # print(sent_tag)
                continue
            opinion_tags = asp_tag.find_all("aspectterm")
            for opinion_tag in opinion_tags:
                term = self.clean_text(opinion_tag.attrs["term"])
                if term not in sent_text: print(sent_text, term)
                polarity = opinion_tag.attrs["polarity"]
                opinion_inst = OpinionInst(term, polarity, None, None, None, None)
                opinion_list.append(opinion_inst)
            sent_Inst = SentInst(sent_id, sent_text, None, None, opinion_list)
            sentence_list.append(sent_Inst)

        return sentence_list

    def clean_str(self, text):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def clean_text(self, text):
        # return word_tokenize(sent_str)
        if self.config.is_stanford_nlp:
            text = self.clean_str(text)
        sent_str = " ".join(text.split("-"))
        sent_str = " ".join(sent_str.split("/"))
        sent_str = " ".join(sent_str.split("("))
        sent_str = " ".join(sent_str.split(")"))
        sent_str = " ".join(sent_str.split(";"))
        sent_str = " ".join(sent_str.split("@"))
        sent_str = " ".join(sent_str.split("#"))
        sent_str = " ".join(sent_str.split())
        return sent_str

    def stanford_tokenize(self, sent_str):
        return self.stanford_nlp.word_tokenize(sent_str)

    def tokenize(self, sent_str):
        '''
        Split a sentence into tokens
        '''
        sent = nlp(sent_str)
        words = [item.text for item in sent]
        tags = [item.pos_ for item in sent]
        return words
        # return tokenizer(sent_str)

    # namedtuple is protected!
    def process_raw_data(self, data, stanford_tokenizer=False):
        '''
        Tokenize each sentence, compute aspect mask for each sentence
        '''
        sent_len = len(data)
        # print('Sentences Num:', sent_len)
        text_words = []  # record all the words
        for sent_i in np.arange(sent_len):
            sent_inst = data[sent_i]
            # Tokenize texts
            if stanford_tokenizer:
                sent_tokens = self.stanford_tokenize(sent_inst.text)
            else:
                sent_tokens = self.tokenize(sent_inst.text)
            text_words.append(sent_tokens)
            sent_inst = sent_inst._replace(text_inds=sent_tokens)
            opinion_list = []
            opi_len = len(sent_inst.opinions)
            # Read  opinion info
            for opi_i in np.arange(opi_len):
                opi_inst = sent_inst.opinions[opi_i]
                target = opi_inst.target_text
                mask = [0] * len(sent_tokens)
                if stanford_tokenizer:
                    target_tokens = self.stanford_tokenize(target)
                else:
                    target_tokens = self.tokenize(target)

                # If no targets specified, skip
                if len(target_tokens) < 1:
                    print('No target')
                    print('Sent:', sent_tokens)
                    continue

                # Find the position of the target, i.e,
                # "I saw the black dog and a white dog stand there, what a lovely black dog, so black dog !"
                sent_tokens = np.array(sent_tokens)
                target_start = target_tokens[0]
                target_end = target_tokens[-1]
                index = np.where(sent_tokens == target_start)[0]
                # Different locations
                for i in index:
                    if i + len(target_tokens) > len(sent_tokens):
                        continue
                    if target_end == sent_tokens[i + len(target_tokens) - 1]:
                        mask[i:(i + len(target_tokens))] = [1] * len(target_tokens)

                if len(index) < 1:
                    print('Target not in the sentence', target_tokens)
                    print('Sentence:', sent_tokens)
                    continue

                label = opi_inst.polarity
                if label == "conflict":  continue  # ignore conflict ones
                # Record label
                opi_inst = opi_inst._replace(class_ind=self.label2id[label])
                # Record target mask
                opi_inst = opi_inst._replace(target_mask=mask)
                # Record tokens
                opi_inst = opi_inst._replace(target_tokens=target_tokens)
                opinion_list.append(opi_inst)
            # update the data
            sent_inst = sent_inst._replace(opinions=opinion_list)
            data[sent_i] = sent_inst
            # print('Raw data tokenized')

        return data, text_words

    def text2ids(self, data, word2id, is_training):
        '''
        Map each word into an id in a text
        Args:
        data: namtuples
        texts: all the text
        '''

        # Build vocab


        def w2id(w):
            try:
                id = word2id[w]
            except:
                id = word2id[self.UNK]
            return id

        # Get the IDs for special tokens
        self.UNK_ID = w2id(self.UNK)
        self.PAD_ID = w2id(self.PAD)
        self.EOS_ID = w2id(self.EOS)
        sent_count = len(data)
        # print('Sentences Num:', sent_len)
        # Update nametuple
        for sent_i in np.arange(sent_count):
            sent_inst = data[sent_i]
            # Tokenize texts
            sent_tokens = sent_inst.text_inds
            # Map each token into an ID
            sent_ids = [w2id(token) for token in sent_tokens]
            sent_inst = sent_inst._replace(text_ids=sent_ids)
            # Read  opinion info
            opi_len = len(sent_inst.opinions)
            # Map target words into IDs
            for opi_i in np.arange(opi_len):
                opi_inst = sent_inst.opinions[opi_i]
                target_tokens = opi_inst.target_tokens
                # target_tokens = self.tokenize(target)
                target_ids = [w2id(token) for token in target_tokens]
                opi_inst = opi_inst._replace(target_ids=target_ids)
                sent_inst.opinions[opi_i] = opi_inst
            data[sent_i] = sent_inst
        return data

    def build_local_vocab(self, texts, max_size):
        '''
        Build and save a vocabulary based on current texts
        texts: lists of words
        '''
        words = []
        for text in texts:
            words.extend(text)
        word_freq_pair = Counter(words)
        print('Tokenized Word Number:', len(word_freq_pair))
        if len(word_freq_pair) < max_size - 3:
            max_size = len(word_freq_pair)

        most_freq = word_freq_pair.most_common(max_size - 3)
        words, _ = zip(*most_freq)
        words = list(words)
        if self.UNK not in words:
            words.insert(0, self.UNK)
        if self.EOS not in words:
            words.append(self.EOS)
        if self.PAD not in words:
            words.append(self.PAD)
        # Build dictionary
        print('Local Vocabulary Size:', len(words))
        word2id = {w: i for i, w in enumerate(words)}
        id2word = {i: w for i, w in enumerate(words)}

        # Save the dictionary
        # dict_file = self.config.dic_path
        # dict_path = os.path.dirname(dict_file)
        # if not os.path.exists(dict_path):
        #     print('Dictionary path doesnot exist')
        #     print('Create...')
        #     os.mkdir(dict_path)
        # with open(dict_file, 'wb') as f:
        #     pickle.dump([word2id, id2word, words], f)
        #     print('Dictionary created successfully')

        return words, [word2id, id2word, words]

    def get_local_word_embeddings(self, pretrained_word_emb, local_vocab):
        '''
        Obtain local word embeddings based on pretrained ones
        local_vocab: word in local vocabulary, in order
        '''
        local_emb = []
        # if the unknow vectors were not given, initialize one
        if self.UNK not in pretrained_word_emb.keys():
            pretrained_word_emb[self.UNK] = np.random.randn(self.config.embed_dim)
        for w in local_vocab:
            local_emb.append(self.word2vec(pretrained_word_emb, w))
        local_emb = np.vstack(local_emb)
        # emb_path = self.config.embed_path
        # if not os.path.exists(os.path.dirname(emb_path)):
        #     print('Path not exists')
        #     os.mkdir(os.path.dirname(emb_path))
        # # Save the local embeddings
        # with open(emb_path, 'wb') as f:
        #     pickle.dump(local_emb, f)
        #     print('Local Embeddings Saved!')
        return local_emb

    def load_pretrained_word_emb(self, file_path):
        '''
        Load a specified vocabulary
        '''
        word_emb = {}
        vocab_words = set()
        with open(file_path) as fi:
            for line in fi:
                items = line.split()
                word = ' '.join(items[:-1 * self.config.embed_dim])
                vec = items[-1 * self.config.embed_dim:]
                word_emb[word] = np.array(vec, dtype=np.float32)
                vocab_words.add(word)
        return word_emb, vocab_words

    def word2vec(self, vocab, word):
        '''
        Map a word into a vec
        '''
        try:
            vec = vocab[word]
        except:
            vec = vocab[self.UNK]
        return vec

    def read(self, data_path, data_format='xml'):
        '''
        read and process raw data, create dictionary and index based on the training data
        '''
        if data_format == 'csv':
            train_data = self.read_csv_data(data_path)
        else:
            train_data = self.read_xml_data(data_path)
        # self.test_data = self.read_xml_data(test_data)
        print('Dataset number:', len(train_data))
        # print('Testing dataset number:', len(self.test_data))
        data = self.process_raw_data(train_data, self.config.is_stanford_nlp)

        # emb = self.load_pretrained_word_emb(config.pretrained_embed_path)
        # _ = self.get_local_word_embeddings(emb, words)
        # test_data = self.process_raw_data(self.test_data)
        return data

    # shuffle and to batch size
    def to_batches(self, data, if_batch=False):
        all_triples = []
        # list of list
        pair_couter = defaultdict(int)
        for sent_inst in data:
            text = sent_inst.text
            tokens = sent_inst.text_inds
            token_ids = sent_inst.text_ids
            # print(tokens)
            for opi_inst in sent_inst.opinions:
                if opi_inst.polarity is None:  continue  # conflict one
                mask = opi_inst.target_mask
                targets = opi_inst.target_tokens
                target_ids = opi_inst.target_ids
                polarity = opi_inst.class_ind
                if tokens is None or mask is None or polarity is None:
                    continue
                all_triples.append([tokens, mask, polarity, token_ids, str(text), targets, target_ids])
                pair_couter[polarity] += 1

        print(pair_couter)
        return all_triples


class DataReader:
    def __init__(self, config, is_training=True):
        '''
        Load dataset and create batches for training and testing
        '''
        self.is_training = is_training
        self.config = config
        self.dh = DataHelper(config)
        self.UNK = self.dh.UNK
        self.EOS = self.dh.EOS
        self.PAD = self.dh.PAD
        self.data = {}
        self.local_emb = None
        self.local_dict = None
        self.word2id = None

    def read_train_test_data(self, data_path_list, data_name, data_source='xml', use_glove=False):
        '''
        Reading Raw Dataset from several files
        Args:
        data_path_list: a list of raw text file paths
        '''
        print('Reading Dataset....')
        data_list = []
        text_word_list = []
        for data_path in data_path_list:
            data, text_words = self.dh.read(data_path, data_source)
            data_list.append(data)
            text_word_list.extend(text_words)
        # Build dictionary based on all the words
        if not use_glove:
            #######################Buld a local vocabulary
            ## Create embeddings for the words in the given dataset
            words, self.local_dict = self.dh.build_local_vocab(text_word_list, self.config.embed_num)
            # Create local embeddings for each word
            # dict_file = self.config.dic_path
            # # The dictionary must be created in advance
            # if not os.path.exists(dict_file):
            #     print('Dictionary file not exist!')
            # with open(dict_file, 'rb') as f:
            #     word2id, _, _ = pickle.load(f)
            emb, _ = self.dh.load_pretrained_word_emb(self.config.pretrained_embed_path)
            self.local_emb = self.dh.get_local_word_embeddings(emb, words)
            self.word2id = {w: i for i, w in enumerate(words)}

        # else:
        #     ####################Use original Glove vocabulary, but the space is large
        #     emb, words = self.dh.load_pretrained_word_emb(self.config.pretrained_embed_path)
        #     # Save word embeddings in binary format
        #     self.local_emb = self.dh.get_local_word_embeddings(emb, words)
        #
        #     # Build dictionary
        #     print('Glove Vocabulary Size:', len(words))
        #     word2id = {w: i for i, w in enumerate(words)}
        # Map words into Ids
        for i, data in enumerate(data_list):
            # Get file name
            name = data_path_list[i]
            data = self.dh.text2ids(data, self.word2id, self.is_training)
            data_batch = self.dh.to_batches(data)
            # Save each processed data
            # self.save_data(data_batch, self.config.data_path + name + '.pkl')
            self.data[name] = data_batch

        # print('Preprocessing Dataset....')
        # self.data_batch = self.dh.to_batches(data)
        # self.data_len = len(self.data_batch)
        self.UNK_ID = self.dh.UNK_ID
        self.PAD_ID = self.dh.PAD_ID
        self.EOS_ID = self.dh.EOS_ID
        self.save_data(self, self.config.pickle_path)
        print('data save to {0}'.format(self.config.pickle_path))
        print('Preprocessing Over!')

    def read_raw_data(self, data_path, data_format='xml', use_glove=False):
        '''
        Reading Raw Dataset from one file
        '''
        print('Reading Dataset....')
        data, text_words = self.dh.read(data_path, data_format)
        # Build dictionary
        if not use_glove:  # Use glove directly, quite large, to be cautious
            words = self.dh.build_local_vocab(text_words, self.config.embed_num)
            # Create local embeddings
            emb, _ = self.dh.load_pretrained_word_emb(self.config.pretrained_embed_path)
            _ = self.dh.get_local_word_embeddings(emb, words)
            # Map words into Ids
            dict_file = self.config.dic_path
            # The dictionary must be created in advance
            if not os.path.exists(dict_file):
                print('Dictionary file not exist!')
            with open(dict_file, 'rb') as f:
                word2id, _, _ = pickle.load(f)
        else:
            ####################Use Glove vocabulary
            emb, words = self.dh.load_pretrained_word_emb(self.config.pretrained_embed_path)
            # Save word embeddings in binary format
            _ = self.dh.get_local_word_embeddings(emb, words)
            # Build dictionary
            print('Glove Vocabulary Size:', len(words))
            word2id = {w: i for i, w in enumerate(words)}
        data = self.dh.text2ids(data, word2id, self.is_training)
        print('Preprocessing Dataset....')
        self.data_batch = self.dh.to_batches(data)
        self.data_len = len(self.data_batch)
        self.UNK_ID = self.dh.UNK_ID
        self.PAD_ID = self.dh.PAD_ID
        self.EOS_ID = self.dh.EOS_ID
        print('Preprocessing Over!')

    def save_data(self, data, save_path):
        '''
        Save the data in specified folder
        '''
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print('Saving successfully!')

    def load_data(self, load_path):
        '''
        Load the dataset
        '''
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.data_batch = pickle.load(f)
                self.data_len = len(self.data_batch)
            self.load_local_dict()
        else:
            print('Data not exist!')
            return None
        return self.data_batch

    def load_local_dict(self):
        '''
        Load dictionary files
        '''
        if not os.path.exists(self.config.dic_path):
            print('Dictionary file not exist!')
        with open(self.config.dic_path, 'rb') as f:
            word2id, _, _ = pickle.load(f)
        self.UNK_ID = word2id[self.dh.UNK]
        self.PAD_ID = word2id[self.dh.PAD]
        self.EOS_ID = word2id[self.dh.EOS]

    def split_save_data(self, train_path, valid_path):
        '''
        split dataset into training and validation parts
        '''
        np.random.shuffle(self.data_batch)
        train_num = int(self.data_len * 5.0 / 6)
        training_batch = self.data_batch[:train_num]
        valid_batch = self.data_batch[train_num:]
        # in case dev too small
        if len(valid_batch) < 500:
            valid_batch = self.data_batch[:500]
            training_batch = self.data_batch[500:]

        try:
            with open(train_path, "wb") as f:
                print('saving to {0}'.format(train_path))
                pickle.dump(training_batch, f)
            with open(valid_path, "wb") as f:
                print('saving to {0}'.format(valid_path))
                pickle.dump(valid_batch, f)
            print('Saving successfully!')
        except:
            print('Saving failure!')


class DataGenerator:
    def __init__(self, data_reader, config, data_batch, is_training=True):
        '''
        Generate training and testing samples
        Args:
        config: configuration parameters
        data_batch: data list, each contain a nametuple
        '''
        self.data_reader = data_reader
        self.is_training = is_training
        self.config = config
        self.index = 0
        # Filter sentences without targets
        self.data_batch = self.remove_empty_target(data_batch)
        self.data_len = len(self.data_batch)
        self.UNK = "unk"
        self.EOS = "<eos>"
        self.PAD = "<pad>"
        self.load_local_dict()

        # options_file = config.elmo_config_file
        # weight_file = config.elmo_weight_file
        # self.elmo = None
        # if options_file != '':
        #     self.elmo = Elmo(options_file, weight_file, 2, dropout=0)

    def remove_empty_target(self, data_batch):
        '''
        Remove items without targets
        '''
        original_num = len(data_batch)
        filtered_data = []
        for item in data_batch:
            if sum(item[1]) > 0:
                filtered_data.append(item)
            else:
                print('Mask Without Target', item[0], 'Target', item[5])
        return filtered_data

    def load_local_dict(self):
        '''
        Load dictionary files
        '''
        # if not os.path.exists(self.config.dic_path):
        #     print('Dictionary file not exist!')
        # with open(self.config.dic_path, 'rb') as f:
        #     word2id, _, _ = pickle.load(f)
        word2id = self.data_reader.word2id

        self.UNK_ID = word2id[self.UNK]
        self.PAD_ID = word2id[self.PAD]
        self.EOS_ID = word2id[self.EOS]

    def generate_sample(self, all_triples):
        '''
        Generate a batch of training dataset
        '''
        batch_size = self.config.batch_size
        select_index = np.random.choice(len(all_triples), batch_size, replace=False)
        select_trip = [all_triples[i] for i in select_index]
        return select_trip

    def generate_balanced_sample(self, all_triples):
        '''
        Generate balanced training data set 
        rate: list, i.e., [0.6, 0.2, 0.2]
        '''
        batch_size = self.config.batch_size
        # labels must be number in order to sort
        labels = [item[2] for item in all_triples]
        unique_label, count_label = np.unique(labels, return_counts=True)
        rate = 1.0 / count_label
        p = [rate[item[2]] for item in all_triples]
        p = p / sum(p)
        select_index = np.random.choice(len(all_triples), batch_size, p=p, replace=False)
        select_trip = [all_triples[i] for i in select_index]
        return select_trip

    def elmo_transform(self, data):
        '''
        Transform sentences into elmo, each sentence represented by words
        '''
        token_list, mask_list, label_list, _, texts, targets, _ = zip(*data)
        sent_lens = [len(tokens) for tokens in token_list]
        sent_lens = torch.LongTensor(sent_lens)
        label_list = torch.LongTensor(label_list)
        max_len = max(sent_lens)
        batch_size = len(sent_lens)
        character_ids = batch_to_ids(token_list)
        embeddings = self.elmo(character_ids)

        # batch_size*word_num * 1024
        sent_vecs = embeddings['elmo_representations'][0]
        sent_vecs = sent_vecs.detach()  # no gradient
        # Padding the mask to same lengths
        mask_vecs = np.zeros([batch_size, max_len])
        mask_vecs = torch.LongTensor(mask_vecs)
        for i, mask in enumerate(mask_list):
            mask_vecs[i, :len(mask)] = torch.LongTensor(mask)
        return sent_vecs, mask_vecs, label_list, sent_lens, texts, targets

    @staticmethod
    def generate(args):

        path_list = [args.train, args.dev, args.test]
        dr = DataReader(args)
        dr.read_train_test_data(path_list, data_name=args.output_name)
        print('Data Preprocessed!')

    @staticmethod
    def load(args):
        data_path = args.pickle_path
        with open(data_path, 'rb') as fh:
            data = pickle.load(fh)

        dg_train = DataGenerator(data, args, data_batch=data.data[args.train], is_training=True)
        dg_dev = DataGenerator(data, args, data_batch=data.data[args.dev], is_training=True)
        dg_test = DataGenerator(data, args, data_batch=data.data[args.test], is_training=True)

        return dg_train, dg_dev, dg_test

    def reset_samples(self):
        self.index = 0

    def pad_data(self, sents, masks, labels, texts, targets, target_ids, sort=True):
        '''
        Padding sentences to same size
        '''
        sent_lens = [len(tokens) for tokens in sents]
        sent_lens = torch.LongTensor(sent_lens)
        label_list = torch.LongTensor(labels)
        max_len = max(sent_lens)
        batch_size = len(sent_lens)
        # Padding mask
        mask_vecs = np.zeros([batch_size, max_len])
        mask_vecs = torch.LongTensor(mask_vecs)
        for i, mask in enumerate(masks):
            mask_vecs[i, :len(mask)] = torch.LongTensor(mask)
        # padding sent with PAD IDs
        sent_vecs = np.ones([batch_size, max_len]) * self.PAD_ID
        sent_vecs = torch.LongTensor(sent_vecs)
        for i, s in enumerate(sents):  # batch_size*max_len
            sent_vecs[i, :len(s)] = torch.LongTensor(s)
        if sort:
            label_list, mask_vecs, sent_ids, sent_lens, target_ids, targets, texts = self.sort_by_length(label_list,
                                                                                                         mask_vecs,
                                                                                                         sent_lens,
                                                                                                         sent_vecs,
                                                                                                         target_ids,
                                                                                                         targets, texts)
        else:
            sent_ids = sent_vecs

        return sent_ids, mask_vecs, label_list, sent_lens, texts, targets, target_ids

    def sort_by_length(self, label_list, mask_vecs, sent_lens, sent_vecs, target_ids, targets, texts):
        sent_lens, perm_idx = sent_lens.sort(0, descending=True)
        sent_ids = sent_vecs[perm_idx]
        mask_vecs = mask_vecs[perm_idx]
        label_list = label_list[perm_idx]
        texts = [texts[i.item()] for i in perm_idx]
        targets = [targets[i.item()] for i in perm_idx]
        target_ids = [target_ids[i.item()] for i in perm_idx]
        return label_list, mask_vecs, sent_ids, sent_lens, target_ids, targets, texts

    def get_ids_samples(self, is_balanced=False, sort=False,pad_target=False):
        '''
        Get samples including ids of words, labels
        '''
        if self.is_training:
            if is_balanced:
                samples = self.generate_balanced_sample(self.data_batch)
            else:
                samples = self.generate_sample(self.data_batch)
            tokens, mask_list, label_list, token_ids, texts, targets, target_ids = zip(*samples)
            # pad both sentence and target
            sent_ids, mask_vecs, label_list, sent_lens, texts, targets, target_ids = self.pad_data(token_ids,
                                                                                          mask_list,
                                                                                          label_list,
                                                                                          texts, targets,
                                                                                          target_ids, sort=sort)
            if pad_target:
                target_ids, _, _, _, _, _, _ = self.pad_data(target_ids,
                                                             target_ids,
                                                             label_list,
                                                             texts, targets,
                                                             target_ids,
                                                             sort=sort)
        else:
            if self.index == self.data_len:
                print('Testing Over!')
            # First get batches of testing data
            if self.data_len - self.index >= self.config.batch_size:
                # print('Testing Sample Index:', self.index)
                start = self.index
                end = start + self.config.batch_size
                samples = self.data_batch[start: end]
                self.index = end
                tokens, mask_list, label_list, token_ids, texts, targets, target_ids = zip(*samples)
                # Sorting happens here
                sent_ids, mask_vecs, label_list, sent_lens, texts, targets, _ = self.pad_data(token_ids,
                                                                                              mask_list,
                                                                                              label_list,
                                                                                              texts, targets,
                                                                                              target_ids, sort=sort)
                if pad_target:
                    target_ids, _, _, _, _, _, _ = self.pad_data(target_ids,
                                                             target_ids,
                                                             label_list,
                                                             texts, targets,
                                                             target_ids,
                                                             sort=sort)

            else:  # Then generate testing data one by one
                samples = self.data_batch[self.index:]
                if self.index == self.data_len - 1:  # if only one sample left
                    samples = [samples]
                tokens, mask_list, label_list, token_ids, texts, targets, target_ids = zip(*samples)
                sent_ids, mask_vecs, label_list, sent_lens, texts, targets, _ = self.pad_data(token_ids,
                                                                                              mask_list,
                                                                                              label_list,
                                                                                              texts, targets,
                                                                                              target_ids, sort=sort)
                if pad_target:
                    target_ids, _, _, _, _, _, _ = self.pad_data(target_ids,
                                                             target_ids,
                                                             label_list,
                                                             texts, targets,
                                                             target_ids,
                                                             sort=sort)
                self.index += len(samples)
        yield sent_ids, mask_vecs, label_list, sent_lens, texts, targets, target_ids

    def get_elmo_samples(self, is_with_texts=False):
        '''
        Generate random samples for training process
        Generate samples for testing process
        sentences represented in Elmo
        '''
        if self.is_training:
            samples = self.generate_sample(self.data_batch)
            sent_vecs, mask_vecs, label_list, sent_lens, texts, targets = self.elmo_transform(samples)
            # Sort the lengths, and change orders accordingly
            sent_lens, perm_idx = sent_lens.sort(0, descending=True)
            sent_vecs = sent_vecs[perm_idx]
            mask_vecs = mask_vecs[perm_idx]
            label_list = label_list[perm_idx]
            texts = [texts[i.item()] for i in perm_idx]
            targets = [targets[i.item()] for i in perm_idx]
        else:
            if self.index == self.data_len:
                print('Testing Over!')
            # First get batches of testing data
            if self.data_len - self.index >= self.config.batch_size:
                # print('Testing Sample Index:', self.index)
                start = self.index
                end = start + self.config.batch_size
                samples = self.data_batch[start: end]
                self.index += self.config.batch_size
                sent_vecs, mask_vecs, label_list, sent_lens, texts, targets = self.elmo_transform(samples)
                # Sort the lengths, and change orders accordingly
                sent_lens, perm_idx = sent_lens.sort(0, descending=True)
                sent_vecs = sent_vecs[perm_idx]
                mask_vecs = mask_vecs[perm_idx]
                label_list = label_list[perm_idx]
                texts = [texts[i.item()] for i in perm_idx]
                targets = [targets[i.item()] for i in perm_idx]
            else:  # Then generate testing data one by one
                samples = self.data_batch[self.index]
                sent_vecs, mask_vecs, label_list, sent_lens, texts, targets = self.elmo_transform([samples])
                self.index += 1
        if is_with_texts:
            yield sent_vecs, mask_vecs, label_list, sent_lens, texts, targets
        else:
            yield sent_vecs, mask_vecs, label_list, sent_lens


# if __name__ == '__main__':
    # DataGenerator.generate(DataConfig())
    # DataGenerator.load(DataConfig())
