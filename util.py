# -*- coding: utf-8 -*-
# @Time    : 17/4/19 10:33 AM
# @Author  : CJ
from torch import optim
import os
import os.path as osp
import torch
import shutil
import yaml
import argparse
import numpy as np
import logging
from sklearn.manifold import TSNE

def load_yaml(file_path,section):
    parser = argparse.ArgumentParser(description='yaml')
    args = parser.parse_args()
    print('Configure ###########')
    with open(file_path) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

    return config


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


def save_checkpoint(state, is_best, filename='', prefix=''):
    torch.save(state, osp.join(filename, '{0}checkpoint.pth.tar'.format(prefix)))
    if is_best:
        shutil.copyfile(osp.join(filename, '{0}checkpoint.pth.tar'.format(prefix)),
                        osp.join(filename, '{0}model_best.pth.tar'.format(prefix)))


def adjust_learning_rate(optimizer, epoch, args):
    '''
    Descend learning rate
    '''
    lr = args.lr / (2 ** (epoch // args.adjust_every))
    # print("Adjust lr to ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def mkdirs(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0

        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

def evaluate_test(dr_test, model, args, sample_out=False):
    mistake_samples = 'data/mistakes.txt'
    with open(mistake_samples, 'w') as f:
        f.write('Test begins...')

    logger.info("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    true_labels = []
    pred_labels = []
    while dr_test.index < dr_test.data_len:
        sent, mask, label, sent_len, texts, targets, _ = next(dr_test.get_ids_samples())
        pred_label = model.predict(sent.cuda(), mask.cuda(), sent_len.cuda())

        # Compute correct predictions
        correct_count += sum(pred_label == label.cuda()).item()

        true_labels.extend(label.cpu().numpy())
        pred_labels.extend(pred_label.cpu().numpy())

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

    acc = correct_count * 1.0 / dr_test.data_len
    f1 = f1_score(true_labels, pred_labels, average='macro')
    print('Confusion Matrix')
    logger.info(confusion_matrix(true_labels, pred_labels))
    logger.info('Accuracy:{}'.format(acc))
    logger.info('f1_score:{}'.format(f1))
    logger.info('precision:{}'.format(precision_score(true_labels, pred_labels, average='macro')))
    logger.info('recall:{}'.format(recall_score(true_labels, pred_labels, average='macro')))
    return acc, f1

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # vec is only 1d vec
    # return the argmax as a python int
    _, idx = torch.max(vec, 0)
    return to_scalar(idx)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0

        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='', prefix=''):
    torch.save(state, osp.join(filename, '{0}checkpoint.pth.tar'.format(prefix)))
    if is_best:
        shutil.copyfile(osp.join(filename, '{0}checkpoint.pth.tar'.format(prefix)),
                        osp.join(filename, '{0}model_best.pth.tar'.format(prefix)))


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(filename)s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


from sklearn.manifold import TSNE
import matplotlib.pylab as plt
def plot_tsne(arr):
    """
        arr = [np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
           np.array([[0, 0, 0], [0, 1, 3], [1, 4, 1], [1, 1, 1]])]
    :param arr:
    :return:
    """
    for i,X in enumerate(arr):
        X_embedded = TSNE(n_components=2).fit_transform(X)
        plt.plot(X_embedded,label=i)
    plt.savefig('tsne.png')