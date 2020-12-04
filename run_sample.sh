mkdir models
mkdir logs

# train bert
python hetmc_run.py --do_train --gradient_accumulation_steps=4 --model_name=test_model --use_bert --train_data_path=./sample_data/train.dialog --test_data_path=./sample_data/test.dialog --bert_model=/path/to/bert_base_chinese1 --max_seq_length=100 --max_dialog_length=80 --train_batch_size=4 --eval_batch_size=4 --num_train_epochs=30 --warmup_proportion=0.1 --patient=50 --learning_rate=5e-5

# train zen
python hetmc_run.py --do_train --gradient_accumulation_steps=4 --model_name=test_model --use_bert --train_data_path=./sample_data/train.dialog --test_data_path=./sample_data/test.dialog --bert_model=/path/to/ZEN_pretrain_base_v0.1.0 --max_seq_length=100 --max_dialog_length=80 --train_batch_size=4 --eval_batch_size=4 --num_train_epochs=30 --warmup_proportion=0.1 --patient=50 --learning_rate=5e-5

# test model
python hetmc_run.py --do_test --test_data_path=./sample_data/test.dialog --eval_model=models/test_model

