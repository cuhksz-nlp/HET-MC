mkdir logs
mkdir models

# train with bert
python hetmc_run.py --do_train --gradient_accumulation_steps=1 --model_name=bert_memory_bilstm_softmax --use_bert --use_memory --train_data_path=./data/train.dialog --test_data_path=./data/test.dialog --bert_model=/path/to/bert_base_chinese1 --max_seq_length=100 --max_dialog_length=80 --train_batch_size=4 --eval_batch_size=4 --num_train_epochs=30 --warmup_proportion=0.1 --patient=50 --learning_rate=1e-5 --utterance_encoder=biLSTM --decoder=softmax --lstm_hidden_size=150

# train with ZEN
python hetmc_run.py --do_train --gradient_accumulation_steps=1 --model_name=bert_memory_bilstm_softmax --use_bert --use_memory --train_data_path=./data/train.dialog --test_data_path=./data/test.dialog --bert_model=/path/to/bert_base_chinese1 --max_seq_length=100 --max_dialog_length=80 --train_batch_size=4 --eval_batch_size=4 --num_train_epochs=30 --warmup_proportion=0.1 --patient=50 --learning_rate=1e-5 --utterance_encoder=biLSTM --decoder=softmax --lstm_hidden_size=150

# test model
python hetmc_run.py --do_test --test_data_path=./data/test.dialog --eval_model=/path/to/model

