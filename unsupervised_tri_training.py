#!/usr/bin/python
from __future__ import division
import matplotlib as mpl

mpl.use('Agg')
import torch
from data_reader_general import data_reader, data_generator
import pickle
import numpy as np
import collections
import codecs
import copy
import os
import models
from collections import Counter
import random
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
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from collections import OrderedDict
import util

import glob
from matplotlib import pyplot as plt

# Get model names in the folder
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
print(model_names)

# Set default parameters of training
parser = argparse.ArgumentParser(description='TSA')
# parser.add_argument('--config', default='cfgs/config_rnn_gcnn_glove.yaml')
parser.add_argument('--config', default='cfgs/indo_tritraining_cnnrnn_random_split.yaml')

parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--e', '--evaluate', action='store_true')

args = parser.parse_args()
RECORDS = collections.defaultdict(list)
DEVICE_NO = 0

# tool functions
def adjust_learning_rate(optimizer, epoch, args):
    '''
    Descend learning rate
    '''
    lr = args.lr / (2 ** (epoch // args.adjust_every))
    print("Adjust lr to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def myplot():
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    plt.rcParams['figure.figsize'] = (8.0, 8.0)  # 设置figure_size尺寸

    def plot_loss():
        plt.subplot(321)
        #### loss
        for key in RECORDS:
            if 'loss_' in key:
                loss = [[(1.0 / len(v) * j + i, np.asscalar(v1)) for j, v1 in enumerate(v)] for i, v in
                        enumerate(RECORDS[key])]
                loss = sum(loss, [])
                loss = pd.DataFrame(loss,
                                    columns=['epoch', 'loss'])
                plt.plot(loss['epoch'], loss['loss'], label=key)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='x-small', bbox_to_anchor=(1, 1),
                   ncol=2)
        # plt.set_ylabel('loss')  # we already handled the x-label with ax1
        # plt.savefig('out/loss.png')

    def plot_acc(idx, label, symbol):
        best_string = {}
        for key in RECORDS:
            if '_metrics' in key:
                data = pd.DataFrame(RECORDS[key])
                plt.subplot(323)
                key = key.split('_')[0]
                for col in data.columns:
                    if 'acc' in col:
                        best = np.round(max(data[col].values), 4)
                        plt.plot(data[col], label='{0}-{1}'.format(key, col))
                        best_string['{0}-{1}'.format(key, col)] = best

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='x-small',
                           bbox_to_anchor=(1, 1), ncol=2)
                plt.subplot(325)
                for col in data.columns:
                    if 'f1' in col:
                        best = np.round(max(data[col].values), 4)
                        plt.plot(data[col], label='{0}-{1}'.format(key, col))
                        best_string['{0}-{1}'.format(key, col)] = best
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='x-small',
                           bbox_to_anchor=(1, 1), ncol=2)

        logger.info('BEST:{0}'.format(json.dumps(best_string, indent=2)))
        # best = np.round(max(data['acc'].values), 4)
        # plt.plot(data['epoch'], data['acc'], symbol, label='{0}:{1}'.format(label, best))

    plot_loss()
    plot_acc(0, 'train accuracy', '-o')

    # plot_acc(2, 'validate accuracy', '-o')
    # plot_acc(4, 'test accuracy', '-o')
    #
    # plot_acc(1, 'train f1', '-*')
    # plot_acc(3, 'validate f1', '-*')
    # plot_acc(5, 'test f1', '-*')
    #

    plt.savefig('out/{0}.png'.format(args.exp_name), dpi=300)


def create_opt(parameters, config):
    '''
    Create optimizer
    '''
    if config.opt == "SGD":
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adam":
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adadelta":
        optimizer = optim.Adadelta(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=config.lr, weight_decay=config.l2)
    return optimizer


def mkdirs(dir):
    '''
    Create folder
    '''
    if not os.path.exists(dir):
        os.mkdir(dir)


def save_checkpoint(save_model, i_iter, args, is_best=True, prefix=''):
    '''
    Save the model to local disk
    '''
    dict_model = save_model.state_dict()
    save_best_checkpoint(dict_model, is_best, args.snapshot_dir, prefix)


def train(model, train, dev, test, optimizer, args, eval_data=None):
    cls_loss_value = AverageMeter(10)
    f_loss_value = AverageMeter(10)
    unsupervised_loss_value = AverageMeter(10)
    best_acc = 0
    best_f1 = 0

    model.train()
    is_best = False
    # weight = torch.Tensor([0.3, 0.59, 0.19])
    # weight = torch.Tensor([0.16, 0.71, 0.13])

    # weight = 1 / weight
    # weight = (weight - torch.min(weight)) / (torch.max(weight) - torch.min(weight))
    # criterion_cls = torch.nn.CrossEntropyLoss(weight=weight.cuda())
    criterion_cls = torch.nn.CrossEntropyLoss()

    logger.info("Start Experiment")
    loops = int(train.data_len / args.batch_size)
    for e_ in range(args.epoch):
        loss_each_epoch = []
        unsuper_loss_dev_each_epoch = []
        unsuper_loss_test_each_epoch = []
        if e_ % args.adjust_every == 0:
            adjust_learning_rate(optimizer, e_, args)

        for idx in range(loops):
            sent_vecs, mask_vecs, label_list, sent_lens, texts, _, _ = next(train.get_ids_samples())
            output1, output2, output3, f_loss = model(sent_vecs.cuda(DEVICE_NO), mask_vecs.cuda(DEVICE_NO),
                                                      label_list.cuda(DEVICE_NO), sent_lens.cuda(DEVICE_NO))
            cls_loss = criterion_cls(output1, label_list.cuda(DEVICE_NO)) + criterion_cls(output2, label_list.cuda(DEVICE_NO))
            cls_loss /= 2
            loss = cls_loss + args.lambda_f * f_loss
            loss_each_epoch.append(loss.data.cpu().numpy())
            cls_loss_value.update(cls_loss.item())
            f_loss_value.update(f_loss.item())
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, norm_type=2)
            optimizer.step()

            if e_ >10:
            # for dev
                model, unsuper_loss = unsupervised(args, criterion_cls, dev, model, optimizer, unsupervised_loss_value)

                if unsuper_loss is not None:
                    unsuper_loss_dev_each_epoch.append(unsuper_loss.data.cpu().numpy())

                # for test
                model, unsuper_loss = unsupervised(args, criterion_cls, test, model, optimizer, unsupervised_loss_value)
                if unsuper_loss is not None:
                    unsuper_loss_test_each_epoch.append(unsuper_loss.data.cpu().numpy())

                # todo:AttributeError: 'Tensor' object has no attribute 'avg'
                # if idx % args.print_freq == 0:
                #     logger.info("i_iter {}/{} cls_loss: {:.3f}\t"
                #                 # "f_lss: {:.5f} \t"
                #                 "unsupervised_loss: {:.5f}".format(idx, loops,
                #                                                    cls_loss_value.avg,
                #                                                    # f_loss_value.avg,
                #
                #                                               unsuper_loss.avg))
        train_metrics = evaluate_test(eval_data[0], model, args)
        valid_metrics = evaluate_test(eval_data[1], model, args)
        test_metrics = evaluate_test(eval_data[2], model, args)

        # RECORDS.append((train_acc, train_f1, valid_acc,
        #                 valid_f1, test_acc, test_f1,
        #                 loss_each_epoch, unsuper_train_acc, unsuper_train_f1, unsuper_valid_acc, unsuper_valid_f1,
        #                 unsuper_loss_dev_each_epoch, unsuper_loss_test_each_epoch))


        valid_acc = valid_metrics['majorvote_acc']
        logger.info("epoch {}, Validation acc: {}".format(e_, valid_acc))
        logger.info("best acc {}".format(best_acc))
        if valid_acc > best_acc:
            is_best = True
            best_acc = valid_acc
            save_checkpoint(model, e_, args, is_best, prefix='acc_')

        # if valid_f1 > best_f1:
        #     is_best = True
        #     best_f1 = valid_f1
        #     save_checkpoint(model, e_, args, is_best, prefix='f1_')
        model.train()
        RECORDS['train_metrics'].append(train_metrics)
        RECORDS['valid_metrics'].append(valid_metrics)
        RECORDS['test_metrics'].append(test_metrics)
        RECORDS['unsuper_loss_dev_each_epoch'].append(unsuper_loss_dev_each_epoch)
        RECORDS['unsuper_loss_test_each_epoch'].append(unsuper_loss_dev_each_epoch)
        RECORDS['loss_each_epoch'].append(loss_each_epoch)
        myplot()
        is_best = False
        # break


def unsupervised(args, criterion_cls, dev, model, optimizer, unsupervised_loss_value):
    sent_vecs_target, mask_vecs_target, label_list_target, sent_lens_target, texts_target, _, _ = next(
        dev.get_ids_samples())
    output1, output2, output3, f_loss = model(sent_vecs_target.cuda(DEVICE_NO), mask_vecs_target.cuda(DEVICE_NO),
                                              label_list_target.cuda(DEVICE_NO), sent_lens_target.cuda(DEVICE_NO))
    output1 = torch.nn.functional.softmax(output1)
    output2 = torch.nn.functional.softmax(output2)
    output1_logit, output1_label = torch.max(output1, dim=1)
    output2_logit, output2_label = torch.max(output2, dim=1)
    output_label_mask = (output1_label == output2_label) & (output1_logit > args.threshold)

    selected_label = output1_label[output_label_mask].view(-1)
    unsupervised_loss = None
    if selected_label.size(0) > 0:
        unsupervised_loss = criterion_cls(output3[output_label_mask],
                                          selected_label)
        unsupervised_loss += criterion_cls(output2[output_label_mask],
                                           selected_label)
        unsupervised_loss += criterion_cls(output1[output_label_mask],
                                           selected_label)
        unsupervised_loss_value.update(unsupervised_loss.item())
        loss = unsupervised_loss * args.unspervised_loss_weight
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, norm_type=2)

        optimizer.step()
    return model, unsupervised_loss


def evaluate_test(dr_test, model, args, sample_out=False):
    mistake_samples = 'data/mistakes.txt'
    with open(mistake_samples, 'w') as f:
        f.write('Test begins...')

    logger.info("Evaluting")
    dr_test.reset_samples()
    model.eval()

    pred_labels = collections.defaultdict(list)
    metrics = {}

    # loops = int(dr_test.data_len / args.batch_size)

    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len, texts, targets, _ = next(dr_test.get_ids_samples())
        output1, output2, output3 = model.predict(sent.cuda(DEVICE_NO), mask.cuda(DEVICE_NO), sent_len.cuda(DEVICE_NO))
        output1 = torch.nn.functional.softmax(output1)
        output2 = torch.nn.functional.softmax(output2)
        output3 = torch.nn.functional.softmax(output3)

        pred_logit3, pred_label3 = torch.max(output3, dim=1)
        # pred_logit3 = pred_logit3.detach().cpu().numpy()
        pred_label3 = pred_label3.cpu().numpy()

        pred_logit2, pred_label2 = torch.max(output2, dim=1)
        # pred_logit2 = pred_logit2.detach().cpu().numpy()
        pred_label2 = pred_label2.cpu().numpy()

        pred_logit1, pred_label1 = torch.max(output1, dim=1)
        pred_logit1 = pred_logit1.detach().cpu().numpy()
        pred_label1 = pred_label1.cpu().numpy()

        pred_labels['pred_label1'].extend(pred_label1)
        pred_labels['pred_label2'].extend(pred_label2)
        pred_labels['pred_label3'].extend(pred_label3)
        pred_labels['true_label'].extend(label.cpu().numpy())

        ## unsupervised_labels
        output_label_mask = (pred_label1 == pred_label2) & (pred_logit1 > args.threshold)
        selected_label = pred_label1[output_label_mask]

        if len(selected_label) > 0:
            pred_labels['unsupervised_true_labels'].extend(label.numpy()[output_label_mask])
            pred_labels['unsupervised_pred_labels'].extend(selected_label)

        ## major vote
        for x in list(zip(pred_label1, pred_label2, pred_label3)):
            pred_label5_counts = Counter(x).most_common()
            if len(pred_label5_counts) < 3:
                pred_labels['majorvote'].append(pred_label5_counts[0][0])
            else:
                pred_labels['majorvote'].append(1)

    unsuper_acc = -0.1
    unsuper_f1 = -0.1
    if len(pred_labels['unsupervised_true_labels']) > 0:
        unsupervised_corr_count = sum(
            [1 for x in list(zip(pred_labels['unsupervised_true_labels'], pred_labels['unsupervised_pred_labels'])) if
             x[0] == x[1]])

        unsuper_acc = unsupervised_corr_count * 1.0 / len(pred_labels['unsupervised_true_labels'])
        print('unsupervised Confusion Matrix')
        print(confusion_matrix(pred_labels['unsupervised_true_labels'], pred_labels['unsupervised_pred_labels']))

        unsuper_f1 = f1_score(pred_labels['unsupervised_true_labels'], pred_labels['unsupervised_pred_labels'],
                              average='macro')

        # f1 = f1_score(true_labels, pred_labels, average='macro')
        print('unsupervised f1_score:', unsuper_f1)
        print('unsupervised acc', unsuper_acc)
    metrics['unsuper_acc'] = unsuper_acc
    metrics['unsuper_f1'] = unsuper_f1

    predictor_names = ['pred_label1', 'pred_label2', 'pred_label3', 'majorvote']
    for key in pred_labels:
        if key in predictor_names:
            corr_count = sum(
                [1 for x in list(zip(pred_labels[key], pred_labels['true_label']))
                 if
                 x[0] == x[1]])
            acc = corr_count * 1.0 / dr_test.data_len
            f1 = f1_score(pred_labels['true_label'], pred_labels[key], average='macro')
            cfm = confusion_matrix(pred_labels['true_label'], pred_labels[key])
            print('{}\nConfusion Matrix: {}'
                  'f1:{},acc:{}'.format(key, cfm, f1, acc
                                        ))
            metrics['{}_acc'.format(key)] = acc
            metrics['{}_f1'.format(key)] = f1
            metrics['{}_cfm'.format(key)] = cfm

            # print("Test Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return metrics


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
    source_train_data = dr.load_data(args.source_path)
    target_train_data = dr.load_data(args.target_train_path)
    test_data = dr.load_data(args.test_path)
    # test_data1 = dr.load_data(args.test_path1)
    logger.info("Training Samples: {}".format(len(source_train_data)))
    logger.info("Validating Samples: {}".format(len(target_train_data)))
    logger.info("Validation Samples: {}".format(len(test_data)))
    # logger.info("Testing Samples: {}".format(len(test_data1)))

    dg_source_train = data_generator(args, source_train_data)
    dg_target_train = data_generator(args, target_train_data)
    dg_test = data_generator(args, test_data)

    dg_source_train_eval = data_generator(args, source_train_data, False)
    dg_target_train_eval = data_generator(args, target_train_data, False)
    dg_test_eval = data_generator(args, test_data, False)

    # dg_test1 = data_generator(args, test_data1, False)

    model = models.__dict__[args.arch](args)
    if args.use_gpu:
        model.cuda(DEVICE_NO)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = create_opt(parameters, args)

    if args.training:
        train(model, dg_source_train, dg_target_train, dg_test, optimizer, args,
              [dg_source_train_eval, dg_target_train_eval, dg_test_eval])
    else:
        pass

    # logger.info("Final Testing Samples: {}".format(len(test_data1)))

    # model_path1 = '{}/model_best.pth.tar'.format(args.snapshot_dir)
    model_path2 = '{}/acc_model_best.pth.tar'.format(args.snapshot_dir)
    model_path3 = '{}/f1_model_best.pth.tar'.format(args.snapshot_dir)
    # model = util.loadModel(model, model_path1)
    # evaluate_test(dg_test1, model, args, sample_out=False)
    # print('best model by accuracy')
    # model = util.loadModel(model, model_path2)
    # # evaluate_test(dg_test1, model, args, sample_out=False)
    # print('best model by f1')
    # model = util.loadModel(model, model_path3)
    # evaluate_test(dg_test1, model, args, sample_out=False)


if __name__ == "__main__":
    main()
