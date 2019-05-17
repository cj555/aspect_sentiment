# 20190517
nohup python train_crf_glove.py --da-grl --exp_name 2 --gpu 1 &
nohup python train_crf_glove.py --da-grl --reverse_da_loss --exp_name 1 --gpu 1 &
nohup python train_crf_glove.py --exp_name 3 --gpu 0 &