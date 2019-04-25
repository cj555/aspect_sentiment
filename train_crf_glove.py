#!/usr/bin/python
from __future__ import division
import matplotlib as mpl

mpl.use('Agg')
import torch
from data_reader_general import data_reader, data_generator
import pickle
import numpy as np
import codecs
import copy
import os
import models
from util import create_logger, AverageMeter
from util import save_checkpoint as save_best_checkpoint
import json
import yaml
from tqdm import tqdm
import os.path as osp
from tensorboardX import SummaryWriter
import logging
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from torch import optim
from sklearn.metrics import classification_report
import pandas as pd
import util
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.cluster import KMeans

# Get model names in the folder
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# Set default parameters of training
parser = argparse.ArgumentParser(description='TSA')
parser.add_argument('--config', default='cfgs/indo_tuning.yaml')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--e', '--evaluate', action='store_true')

args = parser.parse_args()

RECORDS = []


# tool functions
def adjust_learning_rate(optimizer, epoch, args):
    '''
    Descend learning rate
    '''
    lr = args.lr / (2 ** (epoch // args.adjust_every))
    print("Adjust lr to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_opt(parameters, config):
    '''
    Create optimizer
    '''
    if config.opt == "SGD":
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adam":
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adadelta":
        optimizer = optim.Adadelta(parameters, lr=config.lr)
    elif config.opt == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=config.lr)
    return optimizer


def mkdirs(dir):
    '''
    Create folder
    '''
    if not os.path.exists(dir):
        os.mkdir(dir)


def save_checkpoint(save_model, i_iter, args, is_best=True):
    '''
    Save the model to local disk
    '''
    #     suffix = '{}_iter'.format(0)
    dict_model = save_model.state_dict()
    #     print(args.snapshot_dir + suffix)
    filename = args.snapshot_dir
    save_best_checkpoint(dict_model, is_best, filename)


def myplot():
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    def plot_loss():
        #### loss
        loss = [[(1.0 / len(v[6]) * j + i, np.asscalar(v1)) for j, v1 in enumerate(v[6])] for i, v in
                enumerate(RECORDS)]
        loss = sum(loss, [])
        loss = pd.DataFrame(loss,
                            columns=['epoch', 'loss'])
        ax1.plot(loss['epoch'], loss['loss'])
        ax1.set_ylabel('loss')  # we already handled the x-label with ax1
        # plt.savefig('out/loss.png')

    def plot_acc(idx, label, symbol):
        data = pd.DataFrame([(i, v[idx]) for i, v in enumerate(RECORDS)], columns=['epoch', 'acc'])
        best = np.round(max(data['acc'].values), 4)
        ax2.plot(data['epoch'], data['acc'], symbol, label='{0}:{1}'.format(label, best))

    plot_loss()
    plot_acc(0, 'train accuracy', '-o')
    plot_acc(2, 'validate accuracy', '-o')
    plot_acc(4, 'test accuracy', '-o')

    plot_acc(1, 'train f1', '-*')
    plot_acc(3, 'validate f1', '-*')
    plot_acc(5, 'test f1', '-*')

    plt.legend()
    plt.savefig('out/{0}.png'.format(args.exp_name))


def train(model, dg_train, dg_valid, dg_test, optimizer, args, tb_logger, dg_train_cp):
    cls_loss_value = AverageMeter(10)
    best_acc = 0
    model.train()
    is_best = False
    logger.info("Start Experiment")
    loops = int(dg_train.data_len / args.batch_size)
    for e_ in range(args.epoch):
        loss_each_epoch = []
        args.curr_epoch = e_
        if e_ % args.adjust_every == 0:
            adjust_learning_rate(optimizer, e_, args)

        for idx in range(loops):
            sent_vecs, mask_vecs, label_list, sent_lens, _, target_name_list, _ = next(dg_train.get_ids_samples())
            target_name_list = [args.train_targets_cluster_ix[' '.join(x).lower()] for x in target_name_list]
            # target_list = torch.LongTensor(target_list)
            target_name_list = torch.FloatTensor(target_name_list)

            cls_loss = model(sent_vecs.cuda(), mask_vecs.cuda(), label_list.cuda(), sent_lens.cuda(),
                             target_name_list.cuda())
            loss_each_epoch.append(cls_loss.data.cpu().numpy())
            cls_loss_value.update(cls_loss.item())
            model.zero_grad()
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, norm_type=2)
            optimizer.step()

            if idx % args.print_freq == 0:
                logger.info("i_iter {}/{} cls_loss: {:3f}".format(idx, loops, cls_loss_value.avg))
                tb_logger.add_scalar("train_loss", idx + e_ * loops, cls_loss_value.avg)

        # if e_ % 5 == 0:
        valid_acc, valid_f1 = evaluate_test(dg_valid, model, args)
        train_acc, train_f1 = evaluate_test(dg_train_cp, model, args)
        test_acc, test_f1 = evaluate_test(dg_test, model, args)
        logger.info("epoch {}, Validation acc: {}".format(e_, valid_acc))

        if valid_acc > best_acc:
            is_best = True
            best_acc = valid_acc
            save_checkpoint(model, e_, args, is_best)
            output_samples = False
            if e_ % 10 == 0:
                output_samples = True
            test_acc, test_f1 = evaluate_test(dg_test, model, args, output_samples)
            logger.info("epoch {}, Test acc: {}".format(e_, test_acc))
        model.train()
        is_best = False
        RECORDS.append((train_acc, train_f1, valid_acc,
                        valid_f1, test_acc, test_f1,
                        loss_each_epoch))
        myplot()


def evaluate_test(dr_test, model, args, sample_out=False, is_validate=True):
    mistake_samples = 'data/mistakes.txt'
    with open(mistake_samples, 'w') as f:
        f.write('Test begins...')

    logger.info("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    print("transitions matrix ", model.inter_crf.transitions.data)
    all_predictions = []
    all_labels = []
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len, texts, targets, _ = next(dr_test.get_ids_samples())
        pred_label, best_seq = model.predict(sent.cuda(), mask.cuda(), sent_len.cuda())
        # print(pred_label)
        # Compute correct predictions
        correct_count += sum(pred_label == label.cuda()).item()

        ##Output wrong samples, for debugging
        indices = torch.nonzero(pred_label != label.cuda())
        if len(indices) > 0:
            indices = indices.squeeze(1)
        if sample_out:
            with open(mistake_samples, 'a') as f:
                for i in indices:
                    line = texts[i] + '###' + ' '.join(targets[i]) + '###' + str(label[i]) + '###' + str(
                        pred_label[i]) + '\n'
                    f.write(line)

        all_predictions.extend(pred_label)
        all_labels.extend(label)
    print('Confusion Matrix')
    print(confusion_matrix(all_predictions, all_labels))

    acc = correct_count * 1.0 / dr_test.data_len
    print("Test Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    f1 = f1_score(all_predictions, all_labels, average='macro')

    # report = classification_report(y_true=all_labels, y_pred=all_predictions, output_dict=True)
    # report = pd.DataFrame(report).transpose()
    #
    # report.index.name = 'index'
    # report = report.round(3)
    # print("report:\n{0}".format(report))

    # if is_validate:
    #     report.to_csv('{}/eval_validate.csv'.format(args.snapshot_dir))
    # else:
    #     report.to_csv('{}/eval_test.csv'.format(args.snapshot_dir))

    return acc, f1


def main():
    """ Create the model and start the training."""
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)
    mkdirs(osp.join("logs/" + args.exp_name))
    mkdirs(osp.join("checkpoints/" + args.exp_name))
    global logger
    logger = create_logger('global_logger', 'logs/' + args.exp_name + '/log.txt')

    logger.info('{}'.format(args))

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))

    cudnn.enabled = True
    args.snapshot_dir = osp.join(args.snapshot_dir, args.exp_name)

    global tb_logger
    tb_logger = SummaryWriter("logs/" + args.exp_name)
    global best_acc
    best_acc = 0

    ##Load datasets
    dr = data_reader(args)
    args.train_path = args.source_path
    args.valid_path = args.test_path
    args.test_path = args.test_path1

    train_data = dr.load_data(args.train_path)

    args.train_targets = list(set([' '.join(x[5]).lower() for x in train_data]))
    args.target_proxy = [x for x in train_data if len(x[6]) == 1][0][5:7]

    polarities = []
    for t in args.train_targets:
        polarity = [x[2] for x in train_data if ' '.join(x[5]).lower() == t]
        l = len(polarity)
        polarity = Counter(polarity)
        polarity = [polarity[0] / l, polarity[1] / l, polarity[2] / l]
        polarities.append(polarity)

    Kmeans_X = np.array(polarities)
    args.train_prior1 = Kmeans_X
    args.train_targets_cluster_ix = {v: args.train_prior1[i] for i, v in enumerate(args.train_targets)}

    # ncluster = 3
    # kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(Kmeans_X)
    # args.train_targets_cluster_ix = {v:kmeans.labels_[i] for i, v in enumerate(args.train_targets)}
    # args.clusterix2wordix = {}
    # for i in range(ncluster):
    #     target = sorted([k for k in args.train_targets_cluster_ix if args.train_targets_cluster_ix[k]==0])[0]
    #     target_ix = [x[6] for x in train_data if ' '.join(x[5]).lower()==target][0]
    #     args.clusterix2wordix[i] = target_ix


    # args.train_targets_ix = {v:i for i,v in enumerate(args.train_targets)}


    train_data_cp = dr.load_data(args.train_path)
    valid_data = dr.load_data(args.valid_path)
    test_data = dr.load_data(args.test_path)

    ## replace all data target to the same
    # train_data_new = []
    # train_data = format_target_proxy(train_data)
    # train_data_cp = format_target_proxy(train_data_cp)
    # valid_data = format_target_proxy(valid_data)
    # test_data = format_target_proxy(test_data)


    logger.info("Training Samples: {}".format(len(train_data)))
    logger.info("Validating Samples: {}".format(len(valid_data)))
    logger.info("Testing Samples: {}".format(len(test_data)))

    dg_train = data_generator(args, train_data)
    dg_train_cp = data_generator(args, train_data_cp, False)
    dg_valid = data_generator(args, valid_data, False)
    dg_test = data_generator(args, test_data, False)

    model = models.__dict__[args.arch](args)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_opt(parameters, args)

    # twitter_data = dr.load_data('data/tweets/SemEval2017-task4-train.pkl')
    #
    # logger.info("twitter Samples: {}".format(len(twitter_data)))
    # # dg_valid2 = data_generator(args, train_data,False)
    # dg_twitter = data_generator(args, twitter_data)

    if args.use_gpu:
        model.cuda()

    if args.training:
        # args.epoch = args.epoch // 3
        train(model, dg_train, dg_valid, dg_test, optimizer, args, tb_logger, dg_train_cp)

        # model = util.loadModel(model, args.snapshot_dir + '/model_best.pth.tar')
        # args.epoch = args.epoch * 3
        # train(model, dg_train, dg_valid, dg_test, optimizer, args, tb_logger,dg_train_cp)
    else:
        pass

    print('test performance!!!!')
    model = util.loadModel(model, args.snapshot_dir + '/model_best.pth.tar')
    evaluate_test(dg_test, model, args, sample_out=False, is_validate=False)


def format_target_proxy(train_data):
    for d in train_data:
        target_index_start = d[1].index(1)
        target_index_end = target_index_start + len(d[5])
        # sentence = d[0]
        d[0][target_index_start:target_index_end] = args.target_proxy[0]
        d[1][target_index_start:target_index_end] = [1]
        # d.append([x for x in d[5]])
        d[5] = args.target_proxy[0]
        d[6] = args.target_proxy[1]

        # train_data_new.append([d[0],])
    return train_data


if __name__ == "__main__":
    main()
