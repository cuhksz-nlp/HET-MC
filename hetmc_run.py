from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from tqdm import tqdm, trange
from seqeval.metrics import classification_report
from hetmc_helper import read_dialog, get_vocab
from hetmc_eval import Evaluation
from hetmc_model import HET
import datetime


def train(args):

    if args.use_bert and args.use_zen:
        raise ValueError('We cannot use both BERT and ZEN')

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_name = './logs/log-' + now_time
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_file_name,
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    logger.info(vars(args))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.model_name is None:
        raise Warning('model name is not specified, the model will NOT be saved!')
    output_model_dir = os.path.join('./models', args.model_name + '_' + now_time)
    # output_model_dir = os.path.join(args.model_name + '_' + now_time)

    training_data = read_dialog(args.train_data_path)
    testing_data = read_dialog(args.test_data_path)

    label2id, word2id, department2id, disease2id = get_vocab(training_data)

    hpara = HET.init_hyper_parameters(args)
    het_model = HET(word2id, label2id, hpara, model_path=args.bert_model,
                    department2id=department2id, disease2id=disease2id)

    train_examples = het_model.data2example(training_data, flag='train')
    eval_examples = het_model.data2example(testing_data, flag='test')

    num_labels = het_model.num_labels
    convert_examples_to_features = het_model.convert_examples_to_features
    feature2input = het_model.feature2input
    id2label = {label_id: label for label, label_id in het_model.label2id.items()}

    total_params = sum(p.numel() for p in het_model.parameters() if p.requires_grad)
    logger.info('# of trainable parameters: %d' % total_params)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    if args.fp16:
        het_model.half()
    het_model.to(device)

    if n_gpu > 1:
        het_model = torch.nn.DataParallel(het_model)

    param_optimizer = list(het_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        # num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    patient = args.patient

    global_step = 0

    evaluator = Evaluation()

    best_eval = {'SUM1': -1, 'SUM2': -1}
    best_epoch = {'SUM1': -1, 'SUM2': -1}
    best_rouge = {'SUM1': None, 'SUM2': None}
    best_tag_report = {'SUM1': None, 'SUM2': None}
    best_test_rouge = {'SUM1': None, 'SUM2': None}
    best_report = None
    num_of_no_improvement = {'SUM1': 0, 'SUM2': 0}
    results_history = {}

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        np.random.shuffle(train_examples)
        het_model.train()
        tr_loss = 0
        nan_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, start_index in enumerate(tqdm(range(0, len(train_examples), args.train_batch_size))):
            het_model.train()
            batch_examples = train_examples[start_index: min(start_index +
                                                             args.train_batch_size, len(train_examples))]
            if len(batch_examples) == 0:
                continue
            train_features = convert_examples_to_features(batch_examples)
            input_ids, input_mask, l_mask, label_ids, ngram_ids, ngram_positions, segment_ids, valid_ids, \
            lmask, party_mask, party_ids, department_ids, disease_ids = feature2input(device, train_features)

            loss = het_model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask,
                             lmask, party_mask, party_ids, department_ids, disease_ids,
                             ngram_ids, ngram_positions)

            if np.isnan(loss.to('cpu').detach().numpy().any()):
                nan_loss += 1
                logger.info('loss is nan at epoch %d. Times %d' % (epoch, nan_loss))

                if nan_loss > 5:
                    raise ValueError('loss is nan!')
                continue

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        het_model.to(device)

        best_result_update = {'SUM1': False, 'SUM2': False}

        het_model.eval()
        y_true = []
        y_pred = []
        for start_index in range(0, len(eval_examples), args.eval_batch_size):
            eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                                 len(eval_examples))]
            eval_features = convert_examples_to_features(eval_batch_examples)

            input_ids, input_mask, l_mask, label_ids, ngram_ids, ngram_positions, segment_ids, valid_ids, \
            lmask, party_mask, party_ids, department_ids, disease_ids = feature2input(device, eval_features)

            with torch.no_grad():
                tag_seq = het_model(input_ids, segment_ids, input_mask, labels=None,
                                    valid_ids=valid_ids, attention_mask_label=l_mask,
                                    label_mask=lmask, party_mask=party_mask,
                                    party_ids=party_ids, department_ids=department_ids,
                                    disease_ids=disease_ids,
                                    input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

            logits = tag_seq.to('cpu').numpy()
            label_ids = label_ids.to('cpu').numpy()

            for i, example in enumerate(eval_batch_examples):
                temp_1 = []
                temp_2 = []
                if len(example.label) >= args.max_dialog_length:
                    gold_label = example.label[:args.max_dialog_length]
                else:
                    gold_label = example.label
                for j, m in enumerate(gold_label):
                    temp_1.append(m)
                    pred = logits[i][j]
                    if pred == 0:
                        temp_2.append('o')
                    else:
                        temp_2.append(id2label[pred])
                y_true.append(temp_1)
                y_pred.append(temp_2)

        assert len(y_true) == len(y_pred)
        # the evaluation method of cws
        summary_list = [example.summary for example in eval_examples]
        sum1_list = []
        sum2_list = []
        assert len(y_pred) == len(eval_examples)
        for y_pred_item, example in zip(y_pred, eval_examples):
            utterance_list = example.text_a
            sum1 = []
            sum2 = []
            for i, y_pred_label in enumerate(y_pred_item):
                if y_pred_label == '1':
                    sum1.append(utterance_list[i])
                elif y_pred_label == '2':
                    sum2.append(utterance_list[i])
            sum1_list.append('，'.join(sum1))
            sum2_list.append('，'.join(sum2))

        epoch_history = {'prf': None, 'SUM1': None, 'SUM2': None, 'SUM2A': None, 'SUM2B': None}
        y_true_all = []
        y_pred_all = []
        for y_true_item in y_true:
            y_true_all += y_true_item
        for y_pred_item in y_pred:
            y_pred_all += y_pred_item
        # report = classification_report(y_true_all, y_pred_all, digits=4)
        report2 = precision_recall_fscore_support(y_true_all, y_pred_all, labels=['0', '1', '2', 'o'])
        # logger.info(str(report))
        logger.info('dev: epoch\t%d' % epoch)
        str_report2 = '\n'

        sum1_f = None
        sum2_f = None

        for i, ls in enumerate(['0', '1', '2', 'o']):
            p = report2[0][i]
            r = report2[1][i]
            f = report2[2][i]
            if ls == '1':
                sum1_f = f
            if ls == '2':
                sum2_f = f
            str_report2 += '%s\tp: %f\tr: %f\tf: %f\n' % (ls, p, r, f)
        logger.info(str_report2)
        epoch_history['prf'] = str_report2

        sum1_rl = None
        sum2_rl = None

        for target in ['SUM1', 'SUM2']:
            if target == 'SUM1':
                sum_list = sum1_list
            else:
                sum_list = sum2_list
            selected_gold, selceted_pred = evaluator.get_evaluate_list(summary_list, sum_list, target)
            overall_rouge, rouge_list, main_metric_score = evaluator.rouge_score(selceted_pred, selected_gold)
            logger.info('eval: epoch\t%d\t target\t%s' % (epoch, target))
            str_report = '\n'

            for key, value in overall_rouge.items():
                str_report += '%s\t%f\n' % (key, value['f'])
                if key == 'rougeL' and target == 'SUM1':
                    sum1_rl = value['f']
                if key == 'rougeL' and target == 'SUM2':
                    sum2_rl = value['f']
            logger.info(str_report)

            epoch_history[target] = str_report

            if num_of_no_improvement[target] < patient:
                if best_eval[target] < main_metric_score:
                    best_eval[target] = main_metric_score
                    num_of_no_improvement[target] = 0
                    best_epoch[target] = epoch
                    best_rouge[target] = str_report
                    best_tag_report[target] = str_report2
                    # best_report = report
                    best_result_update[target] = True
                else:
                    num_of_no_improvement[target] += 1

        # -------- SUM 2A ----------
        sum_list = sum2_list
        for flag in ['SUM2A', 'SUM2B']:
            selected_gold, selceted_pred = evaluator.get_evaluate_list(summary_list, sum_list, flag)
            overall_rouge, rouge_list, main_metric_score = evaluator.rouge_score(selceted_pred, selected_gold)
            logger.info('eval: epoch\t%d\t target\t%s' % (epoch, flag))
            str_report = ''

            for key, value in overall_rouge.items():
                str_report += '%s\t%f\n' % (key, value['f'])
            logger.info(str_report)

            epoch_history[flag] = str_report
        # -------- SUM 2A ----------

        results_history[epoch] = epoch_history
        # keep
        for target in ['SUM1', 'SUM2']:
            if best_result_update[target]:
                save_model_name = 'model'
                save_model_dir = os.path.join(output_model_dir, save_model_name)

                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)

                model_to_save = het_model.module if hasattr(het_model, 'module') else het_model

                model_to_save.save_model(save_model_dir, args.bert_model)

                with open(os.path.join(save_model_dir, 'test.sum.txt'), "w") as f:
                    for sum1, sum2 in zip(sum1_list, sum2_list):
                        f.write('SUM1\t%s\n' % sum1)
                        f.write('SUM2\t%s\n\n' % sum2)

        if num_of_no_improvement['SUM1'] >= patient and num_of_no_improvement['SUM2'] >= patient:
            logger.info('\nEarly stop triggered at epoch %d\n' % epoch)
            break

    logger.info("\n======= best results ========\n")
    for target in ['SUM1', 'SUM2']:
        logger.info(target + ("\tEpoch: %d\ttest\t" % best_epoch[target]) + str(best_rouge[target])
                    + '\n' + str(best_tag_report[target]))
        logger.info(str(best_report))
    logger.info("\n======= best results ========\n")


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    het_model = HET.load_model(args.eval_model)

    testing_data = read_dialog(args.test_data_path)
    eval_examples = het_model.data2example(testing_data, flag='test')

    convert_examples_to_features = het_model.convert_examples_to_features
    feature2input = het_model.feature2input
    id2label = {label_id: label for label, label_id in het_model.label2id.items()}
    max_dialog_length = het_model.max_dialog_length

    if args.fp16:
        het_model.half()
    het_model.to(device)
    if n_gpu > 1:
        het_model = torch.nn.DataParallel(het_model)

    evaluator = Evaluation()
    het_model.eval()
    y_true = []
    y_pred = []
    for start_index in tqdm(range(0, len(eval_examples), args.eval_batch_size)):
        eval_batch_examples = eval_examples[start_index: min(start_index + args.eval_batch_size,
                                                             len(eval_examples))]
        eval_features = convert_examples_to_features(eval_batch_examples)

        input_ids, input_mask, l_mask, label_ids, ngram_ids, ngram_positions, segment_ids, valid_ids, \
        lmask, party_mask, party_ids, department_ids, disease_ids = feature2input(device, eval_features)

        with torch.no_grad():
            tag_seq = het_model(input_ids, segment_ids, input_mask, labels=None,
                                valid_ids=valid_ids, attention_mask_label=l_mask,
                                label_mask=lmask, party_mask=party_mask,
                                party_ids=party_ids, department_ids=department_ids,
                                disease_ids=disease_ids,
                                input_ngram_ids=ngram_ids, ngram_position_matrix=ngram_positions)

        logits = tag_seq.to('cpu').numpy()

        for i, example in enumerate(eval_batch_examples):
            temp_1 = []
            temp_2 = []
            if len(example.label) >= max_dialog_length:
                gold_label = example.label[:max_dialog_length]
            else:
                gold_label = example.label
            for j, m in enumerate(gold_label):
                temp_1.append(m)
                pred = logits[i][j]
                if pred == 0:
                    temp_2.append('o')
                else:
                    temp_2.append(id2label[pred])
            y_true.append(temp_1)
            y_pred.append(temp_2)

    assert len(y_true) == len(y_pred)
    # the evaluation method of cws
    summary_list = [example.summary for example in eval_examples]
    sum1_list = []
    sum2_list = []
    assert len(y_pred) == len(eval_examples)
    for y_pred_item, example in zip(y_pred, eval_examples):
        utterance_list = example.text_a
        sum1 = []
        sum2 = []
        for i, y_pred_label in enumerate(y_pred_item):
            if y_pred_label == '1':
                sum1.append(utterance_list[i])
            elif y_pred_label == '2':
                sum2.append(utterance_list[i])
        sum1_list.append('，'.join(sum1))
        sum2_list.append('，'.join(sum2))

    with open('test.sum.txt', "w") as f:
        for sum1, sum2 in zip(sum1_list, sum2_list):
            f.write('SUM1\t%s\n' % sum1)
            f.write('SUM2\t%s\n\n' % sum2)

    y_true_all = []
    y_pred_all = []
    for y_true_item in y_true:
        y_true_all += y_true_item
    for y_pred_item in y_pred:
        y_pred_all += y_pred_item
    # report = classification_report(y_true_all, y_pred_all, digits=4)
    print('y_true_len %d' % len(y_true_all))
    print('y_pred_all %d' % len(y_pred_all))
    report2 = precision_recall_fscore_support(y_true_all, y_pred_all, labels=['0', '1', '2', 'o'])

    str_report2 = '\n'

    print(args.test_data_path)

    for i, ls in enumerate(['0', '1', '2', 'o']):
        p = report2[0][i]
        r = report2[1][i]
        f = report2[2][i]
        str_report2 += '%s\tp: %f\tr: %f\tf: %f\n' % (ls, p, r, f)

    print(str_report2)

    for target in ['SUM1', 'SUM2', 'SUM2A']:
        if target == 'SUM1':
            sum_list = sum1_list
        else:
            sum_list = sum2_list
        selected_gold, selceted_pred = evaluator.get_evaluate_list(summary_list, sum_list, target)
        overall_rouge, rouge_list, main_metric_score = evaluator.rouge_score(selceted_pred, selected_gold)
        print(target)
        str_report = '\n'
        for key, value in overall_rouge.items():
            str_report += '%s\t%f\n' % (key, value['f'])
        print(str_report)

    # -------- SUM 2B ----------
    sum_list = sum2_list
    for flag in ['SUM2A', 'SUM2B']:
        selected_gold, selceted_pred = evaluator.get_evaluate_list(summary_list, sum_list, flag, use_subset=True)
        overall_rouge, rouge_list, main_metric_score = evaluator.rouge_score(selceted_pred, selected_gold)
        print('target\t%s' % flag)
        str_report = ''

        for key, value in overall_rouge.items():
            str_report += '%s\t%f\n' % (key, value['f'])
        print(str_report)


def predict(args):
    return


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path. Should contain the .tsv files for the task.")
    parser.add_argument("--test_data_path",
                        default=None,
                        type=str,
                        help="The test data path. Should contain the .tsv files for the task.")
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        help="The data path containing the sentences to be segmented")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        help="The output path of segmented file")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_size",
                        default=128,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patient', type=int, default=3, help="Patient for the early stop.")

    parser.add_argument('--utterance_encoder', type=str, default='biLSTM', help="Can be used for distant debugging.")
    parser.add_argument('--decoder', type=str, default='softmax', help="Can be used for distant debugging.")
    parser.add_argument('--lstm_hidden_size', type=int, default=150, help="Can be used for distant debugging.")
    parser.add_argument('--model_name', type=str, default=None, help="Can be used for distant debugging.")
    parser.add_argument("--max_dialog_length",
                        default=80,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n")
    parser.add_argument("--use_memory",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_party",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_department",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_disease",
                        action='store_true',
                        help="Whether to run training.")
    # parser.add_argument('--eval_flag', type=str, default='',
    #                     help="One of SUM1, SUM2, SUM2A, SUM2B, SUM2_both")

    args = parser.parse_args()

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    elif args.do_predict:
        predict(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    main()
