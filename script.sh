CUDA_VISIBLE_DEVICES=0,1 python run_classifier_TABSA.py \
--task_name semeval_term_NLI_M \
--data_dir data/semeval2014/bert-pair \
--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 12 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--output_dir results/semeval2014/NLI_M \
--seed 42


python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path ~/pretrain/uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file ~/pretrain/uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path ~/pretrain/uncased_L-12_H-768_A-12/pytorch_model.bin


python run_classifier_TABSA.py --task_name semeval_term_NLI_M  --output_dir results/semeval2014/Restaurants/term_NLI_M --data_dir data/semeval2014/bert-pair/Restaurants  --vocab_file uncased_L-12_H-768_A-12/vocab.txt  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json  --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin  --eval_test  --do_lower_case  --max_seq_length 512  --train_batch_size 12  --learning_rate 2e-5  --num_train_epochs 6.0  --seed 42
python run_classifier_TABSA.py --task_name semeval_term_QA_M  --output_dir results/semeval2014/Restaurants/term_QA_M --data_dir data/semeval2014/bert-pair/Restaurants  --vocab_file uncased_L-12_H-768_A-12/vocab.txt  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json  --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin  --eval_test  --do_lower_case  --max_seq_length 512  --train_batch_size 12  --learning_rate 2e-5  --num_train_epochs 6.0  --seed 42
python run_classifier_TABSA.py --task_name semeval_term_NLI_B  --output_dir results/semeval2014/Restaurants/term_NLI_B --data_dir data/semeval2014/bert-pair/Restaurants  --vocab_file uncased_L-12_H-768_A-12/vocab.txt  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json  --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin  --eval_test  --do_lower_case  --max_seq_length 512  --train_batch_size 12  --learning_rate 2e-5  --num_train_epochs 6.0  --seed 42
python run_classifier_TABSA.py --task_name semeval_term_QA_B  --output_dir results/semeval2014/Restaurants/term_QA_B --data_dir data/semeval2014/bert-pair/Restaurants  --vocab_file uncased_L-12_H-768_A-12/vocab.txt  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json  --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin  --eval_test  --do_lower_case  --max_seq_length 512  --train_batch_size 12  --learning_rate 2e-5  --num_train_epochs 6.0  --seed 42

python run_classifier_TABSA.py --task_name semeval_term_NLI_M  --output_dir results/semeval2014/laptop/term_NLI_M --data_dir data/semeval2014/bert-pair/laptop  --vocab_file uncased_L-12_H-768_A-12/vocab.txt  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json  --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin  --eval_test  --do_lower_case  --max_seq_length 512  --train_batch_size 12  --learning_rate 2e-5  --num_train_epochs 6.0  --seed 42
python run_classifier_TABSA.py --task_name semeval_term_QA_M  --output_dir results/semeval2014/laptop/term_QA_M --data_dir data/semeval2014/bert-pair/laptop  --vocab_file uncased_L-12_H-768_A-12/vocab.txt  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json  --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin  --eval_test  --do_lower_case  --max_seq_length 512  --train_batch_size 12  --learning_rate 2e-5  --num_train_epochs 6.0  --seed 42
python run_classifier_TABSA.py --task_name semeval_term_NLI_B  --output_dir results/semeval2014/laptop/term_NLI_B --data_dir data/semeval2014/bert-pair/laptop  --vocab_file uncased_L-12_H-768_A-12/vocab.txt  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json  --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin  --eval_test  --do_lower_case  --max_seq_length 512  --train_batch_size 12  --learning_rate 2e-5  --num_train_epochs 6.0  --seed 42
python run_classifier_TABSA.py --task_name semeval_term_QA_B  --output_dir results/semeval2014/laptop/term_QA_B --data_dir data/semeval2014/bert-pair/laptop  --vocab_file uncased_L-12_H-768_A-12/vocab.txt  --bert_config_file uncased_L-12_H-768_A-12/bert_config.json  --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin  --eval_test  --do_lower_case  --max_seq_length 512  --train_batch_size 12  --learning_rate 2e-5  --num_train_epochs 6.0  --seed 42


python evaluation.py --task_name sem --pred_data_dir results/semeval2014/term_NLI_M/test_ep_1.txt
