#!/usr/bin/python
from __future__ import division
import torch
from data_reader_general import data_reader, data_generator
import pickle
import numpy as np
import codecs
import copy
import os
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
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from model_batch_crf_glove import AspectSent
import glob

# Get model names in the folder
# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# Set default parameters of training
path_tweet = 'cfgs/tweets/config_crf_glove_tweets.yaml'
path_laptop = 'cfgs/laptop/config_crf_cnn_glove_laptop.yaml'
path_res = 'cfgs/config_crf_glove_res.yaml'
path_indo = 'cfgs/indo/config_crf_glove_indo_preprocessed.yaml'
path_eng = 'eng_test.yaml'

files = [path_eng]
parser = argparse.ArgumentParser(description='TSA')
parser.add_argument('--config',
                    default=path_indo)  # 'config_crf_rnn_glove_res.yaml')
parser.add_argument('--pwd', default='', type=str)
parser.add_argument('--e', '--evaluate', action='store_true')
parser.add_argument('--bio', action='store_true')

args = parser.parse_args()


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
    save_best_checkpoint(dict_model, is_best, i_iter, filename)


def train(model, dg_train, dg_valid, dg_test, optimizer, args, tb_logger, dg_da_train, exp, dg_train_eval):
    cls_loss_value = AverageMeter(10)
    best_acc = 0
    best_valid_f1 = 0
    best_train_f1 = 0
    test_f1 = 0
    model.train()
    is_best = False
    logger.info("Start Experiment")

    for e_ in range(args.epoch):
        # logger.info('training sentiment!!')
        for param in model.parameters():
            param.requires_grad = True
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = create_opt(parameters, args)
        if e_ % args.adjust_every == 0:
            adjust_learning_rate(optimizer, e_, args)
        loops = int(dg_train.data_len / args.batch_size)
        for idx in range(loops)[:2]:
            b, a, i = next(dg_train.get_ids_bio_samples())

            # sent_vecs, mask_vecs, label_list, sent_lens, _, _, _, perm_idx = b
            # sent_vecs, mask_vecs, label_list, sent_lens, _, _, _ = next(dg_train.get_ids_samples())

            # if args.if_gpu:
            #     sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
            #     label_list, sent_lens = label_list.cuda(), sent_lens.cuda()
            # cls_loss, norm_pen = model(sent_vecs, mask_vecs, label_list, sent_lens)
            cls_loss, norm_pen = model(b, a, i)

            cls_loss_value.update(cls_loss.item())

            total_loss = cls_loss + norm_pen
            model.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, norm_type=2)
            optimizer.step()

            if idx % args.print_freq == 0:
                model.eval()
                # domain_cls_loss, unsuper_loss = model(sent_vecs, mask_vecs, label_list, sent_lens)

                print(
                    "exp:{}, e_:{}, "
                    "task: sentiment cls, "
                    "sentiment cls loss {:.3f} "
                    "with penalty {:.3f},".format(exp,
                                                  e_,
                                                  cls_loss.item(),
                                                  norm_pen.item()))
                model.train()
                # logger.info("i_iter {}/{} cls_loss: {:3f}".format(idx, loops, cls_loss_value.avg))
                # tb_logger.add_scalar("train_loss", idx + e_ * loops, cls_loss_value.avg)
        model.eval()
        test_f1, valid_f1 = update_test_model(args, best_valid_f1, dg_test, dg_valid, e_, exp, model,
                                              test_f1)
        # train_acc, train_f1 = evaluate_test(dg_train_eval, model, args, False, mode='train')

        # model.train()
        # if valid_f1 < best_valid_f1 and train_f1 > best_train_f1 and args.da == True:
        #     for e1_ in range(args.epoch):
        #         # if e_ % 20 < 15:
        #         model.train()
        #         logger.info('training domain classifier')
        #         for param in model.parameters():
        #             param.requires_grad = False
        #         for param in model.domaincls.parameters():
        #             param.requires_grad = True
        #         for param in model.attn1.parameters():
        #             param.requires_grad = True
        #
        #         parameters = filter(lambda p: p.requires_grad, model.parameters())
        #         optimizer = create_opt(parameters, args)
        #         if e1_ % args.adjust_every == 0:
        #             adjust_learning_rate(optimizer, e1_, args)
        #
        #         loops = int(dg_da_train.data_len / args.batch_size)
        #         for idx in range(loops):
        #             sent_vecs, mask_vecs, label_list, sent_lens, _, _, _ = next(
        #                 dg_da_train.get_ids_samples(is_balanced=True))
        #             if args.if_gpu:
        #                 sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
        #                 label_list, sent_lens = label_list.cuda(), sent_lens.cuda()
        #             domain_cls_loss, unsuper_loss = model(sent_vecs, mask_vecs, label_list, sent_lens,
        #                                                   domain_adapt=True, domain_adapt_mode='cls')
        #             # cls_loss = torch.exp(torch.abs(pair_dis_loss1- pair_dis_loss2))
        #
        #             model.zero_grad()
        #             domain_cls_loss.backward()
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, norm_type=2)
        #             optimizer.step()
        #             if idx % args.print_freq * 20 == 0:
        #                 print("exp:{}, e1_:{}, task:domain cls,"
        #                       "domain_cls_loss:{:.2f}, "
        #                       "da_loss:{:.2f}".format(exp, e1_,
        #                                               domain_cls_loss.item(),
        #                                               unsuper_loss.item()))
        #             # model.eval()
        #             # test_f1 = update_test_model(args, best_f1, dg_test, dg_valid, e_, exp, model, test_f1)
        #             # model.train()
        #
        #     for e1_ in range(args.epoch)[:3]:
        #         # elif e_ % 14 >= 8 and e_ % 14 < 11:
        #         # for e_ in range(args.epoch):
        #         model.train()
        #         logger.info('training embedding classifier')
        #         for param in model.parameters():
        #             param.requires_grad = False
        #         for param in model.bilstm.parameters():
        #             param.requires_grad = True
        #         for param in model.cat_layer.parameters():
        #             param.requires_grad = True
        #
        #         parameters = filter(lambda p: p.requires_grad, model.parameters())
        #         optimizer = create_opt(parameters, args)
        #         if e1_ % args.adjust_every == 0:
        #             adjust_learning_rate(optimizer, e1_, args)
        #         loops = int(dg_da_train.data_len / args.batch_size)
        #         for idx in range(loops):
        #             sent_vecs, mask_vecs, label_list, sent_lens, _, _, _ = next(
        #                 dg_da_train.get_ids_samples(is_balanced=True))
        #             if args.if_gpu:
        #                 sent_vecs, mask_vecs = sent_vecs.cuda(), mask_vecs.cuda()
        #                 label_list, sent_lens = label_list.cuda(), sent_lens.cuda()
        #             domain_cls_loss, unsuper_loss = model(sent_vecs, mask_vecs, label_list, sent_lens,
        #                                                   domain_adapt=True, domain_adapt_mode='cls')
        #             # cls_loss = torch.exp(torch.abs(pair_dis_loss1- pair_dis_loss2))
        #             if unsuper_loss > 0:
        #                 model.zero_grad()
        #                 unsuper_loss.backward()
        #                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, norm_type=2)
        #                 optimizer.step()
        #             if idx % args.print_freq == 0:
        #                 print(
        #                     "exp:{}, e1_:{}, task:domain align,"
        #                     "domain_cls_loss:{:.2f}, "
        #                     "da_loss:{:.2f}".format(exp, e1_,
        #                                             domain_cls_loss.item(),
        #                                             unsuper_loss.item()))
        #         # model.eval()
        #
        # best_train_f1 = max(train_f1, best_train_f1)
        # best_valid_f1 = max(valid_f1, best_valid_f1)
        # logger.info("Best Test f1_score: {}".format(test_f1))

        # test_f1 = update_test_model(args, best_f1, dg_test, dg_valid, e_, exp, model, test_f1)
        # model.train()


def update_test_model(args, best_valid_f1, dg_test, dg_valid, e_, exp, model, test_f1):
    valid_acc, valid_f1 = evaluate_test(dg_valid, model, args, mode='valid')
    logger.info("epoch {}, Validation f1: {}".format(e_, valid_f1))
    # if valid_acc > best_acc:
    #     is_best = True
    #     best_acc = valid_acc
    #     save_checkpoint(model, e_, args, is_best)
    #     output_samples = False
    #     if e_ % 10 == 0:
    #         output_samples = True
    #     test_acc, test_f1 = evaluate_test(dg_test, model, args, output_samples)
    #     logger.info("epoch {}, Test acc: {}".format(e_, test_acc))
    if valid_f1 > best_valid_f1:
        is_best = True
        save_checkpoint(model, e_, args, is_best)
        output_samples = False
        if e_ % 10 == 0:
            output_samples = True
        test_acc, test_f1 = evaluate_test(dg_test, model, args, output_samples, mode='test')
        logger.info("exp{}, epoch {}, Test f1_score: {}".format(exp, e_, test_f1))

        model.train()
        is_best = False
    return test_f1, valid_f1


def evaluate_test(dr_test, model, args, sample_out=False, mode='valid'):
    mistake_samples = '{0}_mistakes.txt'.format(args.exp_name)
    with open(mistake_samples, 'a') as f:
        f.write('Test begins...')

    logger.info("Evaluting")
    dr_test.reset_samples()
    model.eval()
    all_counter = 0
    correct_count = 0
    true_labels = []
    pred_labels = []
    print("transitions matrix ", model.inter_crf.transitions.data)
    while dr_test.index < dr_test.data_len:
        b, a, i = next(dr_test.get_ids_bio_samples())
        # sent, mask, label, sent_len, texts, targets, _, perm_idx = b
        # if args.if_gpu:
        #     sent, mask, sent_len, label = sent.cuda(), mask.cuda(), sent_len.cuda(), label.cuda()
        # pred_label, best_seq = model.predict(sent, mask, sent_len)

        pred_label, best_seq = model.predict(b, a, i)
        # Compute correct predictions
        correct_count += sum(pred_label == label).item()

        true_labels.extend(label.cpu().numpy())
        pred_labels.extend(pred_label.cpu().numpy())

        ##Output wrong samples, for debugging
        indices = torch.nonzero(pred_label != label)
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
    # logger.info('Confusion Matrix:')
    # logger.info(confusion_matrix(true_labels, pred_labels))
    logger.info('{}|Acc:{:.3f},'
                'f1:{:.3f},'
                'precison:{:.3f},'
                'recall:{:.3f}'.format(mode, acc, f1, precision_score(true_labels, pred_labels, average='macro'),
                                       recall_score(true_labels, pred_labels, average='macro')))
    # logger.info('f1_score:{}'.format(f1))
    # logger.info('precision:{}'.format()
    # logger.info('recall:{}'.format())

    #     print('Confusion Matrix')
    #     print(confusion_matrix(true_labels, pred_labels))
    #     print('f1_score:', f1_score(true_labels, pred_labels, average='macro'))
    #     print("Sentiment Accuray {0}, {1}:{2}".format(acc, correct_count, all_counter))
    return acc, f1


def main(train_path, valid_path, test_path, exp=0):
    """ Create the model and start the training."""

    for file in files:
        load_config(file)
        global logger
        logger = create_logger('global_logger', 'logs/' + args.exp_name + '/log.txt')
        logger.info('{}'.format(args))
        logger.info(
            '============Exp:{3}\ntraining:{0}\nvalid:{1}\ntest:{2}'.format(train_path, valid_path, test_path, exp))

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
        args.train_path = train_path
        args.valid_path = valid_path
        args.test_path = test_path
        train_data = dr.load_data(args.train_path)

        if valid_path == test_path:
            tmp_data = dr.load_data(args.test_path)
            valid_num = int(len(tmp_data) * 0.1)
            valid_data = copy.deepcopy(tmp_data[:valid_num])
            test_data = copy.deepcopy(tmp_data[valid_num:])

        else:
            valid_data = dr.load_data(args.valid_path)
            test_data = dr.load_data(args.test_path)

        da_train_data = copy.deepcopy(train_data) + copy.deepcopy(valid_data) + copy.deepcopy(test_data)
        for idx, datum in enumerate(da_train_data):
            if idx < len(train_data):
                datum[2] = 0
            elif idx >= len(train_data) and idx < len(valid_data) + len(train_data):
                datum[2] = 1
            else:
                datum[2] = 2

        # train_data = dr.load_data(train_path)
        # valid_data = dr.load_data(valid_path)
        # test_data = dr.load_data(test_path)

        logger.info("Training Samples: {}".format(len(train_data)))
        logger.info("Validating Samples: {}".format(len(valid_data)))
        logger.info("Testing Samples: {}".format(len(test_data)))

        dg_train = data_generator(args, train_data)
        dg_train_da = data_generator(args, da_train_data)
        dg_valid = data_generator(args, valid_data, False)
        dg_test = data_generator(args, test_data, False)
        dg_train_eval = data_generator(args, copy.deepcopy(train_data), False)

        # model = models.__dict__[args.arch](args)
        model = AspectSent(args)

        path = None  # 'checkpoints/config_crf_glove_tweets_20181206_3/checkpoint.pth.tar9'
        if path:
            model.load_state_dict(torch.load(path))
        if args.if_gpu:
            model = model.cuda()
        # parameters = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = create_opt(parameters, args)

        if args.training:
            train(model, dg_train, dg_valid, dg_test, None, args, tb_logger, dg_train_da, exp, dg_train_eval)
        else:
            print('NOT arg.training')
            PATH = "checkpoints/config_crf_glove_tweets_20190212/checkpoint.pth.tar21"
            model.load_state_dict(torch.load(PATH))
            evaluate_test(dg_test, model, args, sample_out=False, mode='test')
        logger.info('============Exp Done:{3}\ntraining:{0}\nvalid:{1}\ntest:{2}'.format(traf, valid, test, exp))


def load_config(file):
    with open(file) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
    mkdirs(osp.join("logs/" + args.exp_name))
    mkdirs(osp.join("checkpoints/" + args.exp_name))


if __name__ == "__main__":

    data_path = 'eng_glove_preprocessed/'
    test_fi = [x for x in glob.glob(('{0}/processed_*test*').format(data_path))]
    train_fi = [x for x in glob.glob(('{0}/processed_*train*').format(data_path))]
    exp = 0
    for traf in train_fi:
        train_key = traf.split('/')[-1].split('_')[1]
        for valid in test_fi:
            valid_key = valid.split('/')[-1].split('_')[1]
            for test in test_fi:
                test_key = test.split('/')[-1].split('_')[1]
                # if valid != test and train_key != test_key and train_key != valid_key and valid_key != train_key:
                if valid_key == test_key:
                    exp += 1
                    main(train_path=traf, valid_path=valid, test_path=test, exp=exp)

    pass

    # main()
