from data_reader_general import *
import matplotlib as mpl
mpl.use('Agg')

import argparse
import yaml
import numpy as np
import pandas as pd
import pickle
from data_reader_general import data_reader, data_generator

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt



# Set default parameters of preprocessing data
parser = argparse.ArgumentParser(description='TSA')
parser.add_argument('--config', default='cfgs/config_crf_glove.yaml')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--e', '--evaluate', action='store_true')

args = parser.parse_args()


def train():
    # Load configuration file
    args.config = '/home/juan_cheng/nlp_tutorial/SA_Sent_trilearning/new_cfgs/semi_supervised_tri_training_exp006.yaml'
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

        #############Load and process raw files########
        #     path1 = "data/laptop/Laptop_Train_v2.xml"
        #     path2 = "data/laptop/Laptops_Test_Gold.xml"
        #     path_list = [path1, path2]
        #     #First time, need to preprocess and save the data
        #     #Read XML file
        #     dr = data_reader(args)
        #     dr.read_train_test_data(path_list)
        #     print('Data Preprocessed!')
        #
        #
        # #     ###############Load preprocessed files, split training and dev parts if necessary#########
        #     dr = data_reader(args)
        #     data = dr.load_data('data/laptop/Laptop_Train_v2.xml.pkl')
        #     dr.split_save_data(args.train_path, args.valid_path)
        #     print('Splitting finished')

    path1 = "data/indo0/train.xml"
    path2 = "data/indo0/dev.xml"
    path3 = "data/indo0/test.xml"
    args.pretrained_embed_path = 'data/glove_indo_original.100d.txt'
    path_list = [path1, path2, path3]

    # path3 = "/home/juan_cheng/nlp_tutorial/SA_Sent_trilearning/data/indo0/test.xml"
    # path_list = [path3]
    # First time, need to preprocess and save the data
    # Read XML file

    dr = data_reader(args)
    # dr.read_train_test_data(path_list,swap_target_dict = get_target_swap_pair())
    dr.read_train_test_data(path_list,data_name = 'target_split')


    print('Data Preprocessed!')


    # #     ###############Load preprocessed files, split training and dev parts if necessary#########
    # dr = data_reader(args)
    # data = dr.load_data('data/laptop/Laptop_Train_v2.xml.pkl')
    # dr.split_save_data(args.train_path, args.valid_path)
    # print('Splitting finished')


def train_csv():
    # Load configuration file
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

    #############Load and process raw files########

    args.data_path = 'data/tweets/'
    args.test_path = 'data/tweets/SemEval2017-task4-test.pkl'
    args.valid_path = 'data/tweets/SemEval2017-task4-valid.pkl'
    args.train_path = 'data/tweets/SemEval2017-task4-train.pkl'
    args.pretrained_embed_path = '/home/chen_juan/pretrain/glove.840B.300d.txt'
    path_raw = "data/tweets/SemEval2017-task4.txt"
    path1 = "data/tweets/SemEval2017-task4-format.csv"
    df = pd.read_csv(path_raw, sep='\t', header=None)[[1, 2]]
    df.columns = ['label', 'text']
    df.dropna(inplace=True)
    df['text'] = [str(x) for x in df['text'].values]
    print(df.groupby('label').count())
    df['target'] = [x.split(" ")[0] for x in df['text'].values]
    df.to_csv(path_or_buf=path1, index=False)
    path_list = [path1]
    # First time, need to preprocess and save the data
    # Read XML file

    dr = data_reader(args)
    dr.read_train_test_data(path_list, data_source='csv', use_glove=True)
    print('Data Preprocessed!')

    #     ###############Load preprocessed files, split training and dev parts if necessary#########
    dr = data_reader(args)
    data = dr.load_data(path1 + '.pkl')
    dr.split_save_data(args.train_path, args.valid_path)
    print('Splitting finished')


def data_desp():

    # train = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo0/train.xml.pkl'
    # dev = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo0/dev.xml.pkl'
    # test = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo0/test.xml.pkl'
    #

    source = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo0/source.pkl'
    target_train_20 = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo0/supervised_target_train_20.pkl'
    target_train = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo0/target_train.pkl'
    target_dev = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo0/target_dev.pkl'
    test = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo0/test.xml.pkl'
    #


    # train = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/random_train.pkl'
    # dev = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/random_dev.pkl'
    # test = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/random_test.pkl'

    # train = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/train.xml.pkl'
    # dev = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/dev.xml.pkl'
    # test = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/test.xml.pkl'

    # train = 'data/indo/source.pkl'
    # dev = 'data/indo/target_dev.pkl'
    # test = 'data/indo/test.xml.pkl'

    # train = 'dso_tsa/Indonesian/target_split/processed_indo_xml_tweets/train.xml.pkl'
    # dev = 'dso_tsa/Indonesian/target_split/processed_indo_xml_tweets/dev.xml.pkl'
    # test = 'dso_tsa/Indonesian/target_split/processed_indo_xml_tweets/test.xml.pkl'

    # train = 'dso_tsa/Indonesian/random_split/shuffled_train.pkl'
    # dev = 'dso_tsa/Indonesian/random_split/shuffled_dev.pkl'
    # test = 'dso_tsa/Indonesian/random_split/shuffled_test.pkl'


    # train = 'dso_tsa/English/dong/train.pkl'
    # dev = 'dso_tsa/English/dong/valid.pkl'
    # test = 'dso_tsa/English/dong/test.csv.pkl'


    # train = 'dso_tsa/English/dong/split_by_target/new_train.pkl'
    # dev = 'dso_tsa/English/dong/split_by_target/new_dev.pkl'
    # test = 'dso_tsa/English/dong/split_by_target/new_test.pkl'

    results = []
    # for fn in [train, dev, test]:
    for fn in [source, target_train, target_train_20, target_dev,test]:
        with open(fn, 'rb') as fh:
            data = pickle.load(fh)
            dn = fn.split('/')[-1].split('.')[0]
            # results[fn.split('/')[-1].split('.')[0]] = pd.DataFrame([(datum[2],' '.join(datum[5]).lower()) for datum in data],columns = ['label','target']).groupby(['target','label']).size().unstack('label', fill_value=0)
            results.extend([(dn, datum[2], ' '.join(datum[5]).lower()) for datum in data])

    return pd.DataFrame(results, columns=['dataset', 'label', 'target']).groupby(['dataset', 'target', 'label']).size().unstack(['label', 'dataset'], fill_value=0).sort_values(by=[(0,'source')],ascending=False).replace(0,'-')



def spilt():
    from random import shuffle
    train = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/target_train.pkl'
    dev = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/target_dev.pkl'
    test = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/target_test.pkl'

    # train = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/random_train.pkl'
    # dev = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/random_dev.pkl'
    # test = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/random_test.pkl'

    data = []
    size = []
    for fn in [train, dev, test]:
        with open(fn, 'rb') as fh:
            datum = pickle.load(fh)
            size.append(len(datum))
            data.extend(datum)
    print('target split size:{0}'.format(size))
    shuffle(data)
    for fn in ['random_train.pkl', 'random_dev.pkl', 'random_test.pkl']:
        rng = None
        if 'train' in fn:
            rng = (0, size[0])
        elif 'dev' in fn:
            rng = (size[0], size[0] + size[1])
        else:
            rng = (size[0] + size[1], size[0] + size[1]+ size[2])
        fn = '/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo/' + fn

        with open(fn, 'wb') as fh:
            pickle.dump(data[rng[0]:rng[1]], fh)

        with open(fn, 'rb') as fh:
            print(fn + ":{0}".format(len(pickle.load(fh))))


def get_target_swap_pair():
    return {
        'djarot':'pak ahok',
        'bu sri':'bu susi',
        'obama': 'lee hsien loong',
        'trump': 'lee hsien loong',
        'bu mega':'setnov',
        'prabowo':'setnov',
        'sby':'setnov',
        'jokowi':'pak kalla',
        'al qaeda':'daesh'
    }

def plot_tsne(arr, name):
    # arr = [np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
    #        np.array([[0, 0, 0], [0, 1, 3], [1, 4, 1], [1, 1, 1]])
    #        ]

    for i, X in enumerate(arr):
        X_embedded = TSNE(n_components=2).fit_transform(X)
        plt.plot(X_embedded[:,0],X_embedded[:,1],'o',label = str(i))

    plt.legend()
    plt.savefig('out/{0}.png'.format(name))


def vis_word_embeding():

    dr = data_reader(args)

    args.embed_path ='data/indo0/vocab/local_emb.pkl'
    args.data_path = ' data/indo0/'
    args.dic_path = 'data/indo0/vocab/dict.pkl'
    #
    # target split
    # train_data = dr.load_data('data/indo0/train.xml.pkl')
    # valid_data = dr.load_data('data/indo0/dev.xml.pkl')
    # test_data = dr.load_data('data/indo0/test.xml.pkl')


    # random split
    train_data = dr.load_data('data/indo0/random10_keep/random_train.pkl')
    valid_data = dr.load_data('data/indo0/random10_keep/random_dev.pkl')
    test_data = dr.load_data('data/indo0/random10_keep/random_test.pkl')



    with open(args.embed_path,'rb') as fh:
        embed = pickle.load(fh)

    train_words = [embed[i] for i in list(set(sum([x[3] for x in train_data], [])))]
    valid_words =  [embed[i] for i in list(set(sum([x[3] for x in valid_data], [])))]
    dev_words = [embed[i] for i in list(set(sum([x[3] for x in test_data], [])))]

    arr = [train_words,valid_words,dev_words]
    plot_tsne(arr, 'tsne')

if __name__ == "__main__":
    with open('data/indo/target_split.pkl','rb') as fh:
        data = pickle.load(fh)
        pass
    # vis_word_embeding()
    # train()
    # train_csv()
    # spilt()
    # data_desp().to_csv('/home/chen_juan/nlp_tutorial/SA_Sent_trilearning/data/indo0/source_target_test.csv')
    # train()
    # spilt()
