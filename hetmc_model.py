from __future__ import absolute_import, division, print_function

import math
import os
import numpy as np
import torch
import subprocess
from torch import nn
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

import pytorch_pretrained_zen as zen
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.crf import CRF

from hetmc_helper import load_json, save_json, read_dialog

DEFAULT_HPARA = {
    'max_seq_length': 128,
    'max_ngram_size': 128,
    'use_bert': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_memory': False,
    'use_party': False,
    'use_department': False,
    'use_disease': False,
    'utterance_encoder': 'biLSTM',
    'decoder': 'softmax',
    'lstm_hidden_size': 150,
    'max_dialog_length': 80
}


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class Memory(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Memory, self).__init__()

        self.temper = hidden_size ** 0.5
        self.hidden_size = hidden_size
        # self.word_embedding_a = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_embedding_c = nn.Embedding(vocab_size, hidden_size)
        # self.linear_1 = nn.Linear(config.word_embedding_dim, config.hidden_size, bias=False)
        # self.linear_2 = nn.Linear(config.word_embedding_dim, 64)
        self.memory_encoder = nn.LSTM(input_size=hidden_size, hidden_size=int(hidden_size / 2),
                                      bidirectional=True, batch_first=True)
        self.layer_norm = LayerNormalization(hidden_size)

    def memory_embeddings(self, input_ids):
        # input_ids: (batch_size * dialog_length, word_length)

        # word_embedding_a: (batch_size * dialog_length, word_length, hidden_size)
        # word_embedding_a = self.word_embedding_a(input_ids)
        word_embedding_c = self.word_embedding_c(input_ids)
        self.memory_encoder.flatten_parameters()
        word_embedding_c, _ = self.memory_encoder(word_embedding_c)
        word_embedding_c = word_embedding_c[:, -1, :]

        # word_embedding_c = self.layer_norm(word_embedding_c)

        return word_embedding_c

    def forward(self, embedding_c, hidden_state, party_mask_metrix):
        # word_seq: (batch_size, word_seq_len)
        # hidden_state: (batch_size, character_seq_len, hidden_size)
        # mask_matrix: (batch_size, character_seq_len, word_seq_len)
        # embedding (batch_size, word_seq_len, hidden_size)

        # embedding_a = self.word_embedding_a(word_seq)
        # embedding_c: (batch_size, word_seq_len, hidden_size)
        # embedding_c = self.word_embedding_c(label_value_matrix)

        tmp_hidden_state = hidden_state.permute(0, 2, 1)
        # u: (batch_size, character_seq_len, word_seq_len)
        # u = torch.matmul(hidden_state, tmp_hidden_state) / self.temper
        u = torch.matmul(hidden_state, tmp_hidden_state) / self.hidden_size

        # print('u shape:', u.shape)

        # p (batch_size, character_seq_len, word_seq_len)
        party_mask_metrix = torch.clamp(party_mask_metrix, 0, 1)

        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, party_mask_metrix)

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)

        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        # character_attention (batch_size, character_seq_len, hidden_state)
        # o = torch.sum(o, 2)
        o = torch.bmm(p, embedding_c)

        return o


class HET(nn.Module):

    def __init__(self, word2id, label2id, hpara, model_path, department2id=None, disease2id=None):
        super().__init__()

        self.word2id = word2id
        self.department2id = None
        self.disease2id = None
        self.label2id = label2id
        self.party2id = None
        self.hpara = hpara
        self.num_labels = len(self.label2id)
        self.max_seq_length = self.hpara['max_seq_length']
        self.use_memory = self.hpara['use_memory']
        self.use_department = self.hpara['use_department']
        self.use_party = self.hpara['use_party']
        self.use_disease = self.hpara['use_disease']
        self.decoder = self.hpara['decoder']
        self.lstm_hidden_size = self.hpara['lstm_hidden_size']
        self.max_dialog_length = self.hpara['max_dialog_length']

        self.bert_tokenizer = None
        self.bert = None
        self.zen_tokenizer = None
        self.zen = None
        self.zen_ngram_dict = None

        if self.hpara['use_bert']:
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_zen']:
            self.zen_tokenizer = zen.BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.zen_ngram_dict = zen.ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = zen.modeling.ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        ori_hidden_size = hidden_size

        if self.use_memory:
            self.memory = Memory(hidden_size, len(word2id))
            hidden_size = hidden_size * 2
        else:
            self.memory = None

        if self.use_party:
            self.party_embedding = nn.Embedding(5, ori_hidden_size)
            hidden_size += ori_hidden_size
            self.party2id = {'<PAD>': 0, '<UNK>': 1, 'P': 2, 'D': 3}
        else:
            self.party_embedding = None

        utterance_hidden_size = hidden_size
        if self.hpara['utterance_encoder'] == 'LSTM':
            self.utterance_encoder = nn.LSTM(input_size=hidden_size, hidden_size=self.lstm_hidden_size,
                                             bidirectional=False, batch_first=True)
            utterance_hidden_size = self.lstm_hidden_size
        elif self.hpara['utterance_encoder'] == 'biLSTM':
            self.utterance_encoder = nn.LSTM(input_size=hidden_size, hidden_size=self.lstm_hidden_size,
                                             bidirectional=True, batch_first=True)
            utterance_hidden_size = self.lstm_hidden_size * 2
        else:
            self.utterance_encoder = None

        if self.use_department:
            self.department_embedding = nn.Embedding(len(department2id), utterance_hidden_size)
            self.department2id = department2id
        else:
            self.department_embedding = None

        if self.use_disease:
            self.disease_embedding = nn.Embedding(len(disease2id), utterance_hidden_size)
            self.disease2id = disease2id
        else:
            self.disease_embedding = None

        if self.use_department and self.use_disease:
            utterance_hidden_size = utterance_hidden_size * 3
        elif self.use_department or self.use_disease:
            utterance_hidden_size = utterance_hidden_size * 2

        self.classifier = nn.Linear(utterance_hidden_size, self.num_labels)

        if self.decoder == 'softmax':
            self.loss_fct = CrossEntropyLoss(ignore_index=0)
        elif self.decoder == 'crf':
            self.crf = CRF(self.num_labels, batch_first=True)
        else:
            raise ValueError()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, label_mask=None,
                party_mask=None, party_ids=None, department_ids=None, disease_ids=None,
                input_ngram_ids=None, ngram_position_matrix=None):

        batch_size = input_ids.shape[0]
        dialog_length = input_ids.shape[1]
        utterance_length = input_ids.shape[2]

        input_ids = input_ids.view(batch_size * dialog_length, utterance_length)
        token_type_ids = token_type_ids.view(batch_size * dialog_length, utterance_length)
        attention_mask = attention_mask.view(batch_size * dialog_length, utterance_length)

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.zen is not None:
            ngram_position_matrix = ngram_position_matrix.view(batch_size * dialog_length, utterance_length, -1)
            input_ngram_ids = input_ngram_ids.view(batch_size * dialog_length, -1)
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()

        word_embedding_c = None
        if self.use_memory:
            word_embedding_c = self.memory.memory_embeddings(input_ids)

        tmp_sequence_output = sequence_output.view(batch_size, dialog_length, utterance_length, -1)
        # word_embedding_a = word_embedding_a.view(batch_size, dialog_length, -1)

        sequence_output = tmp_sequence_output[:, :, 0]
        tmp_label_mask = torch.stack([label_mask] * sequence_output.shape[-1], 2)
        sequence_output = torch.mul(sequence_output, tmp_label_mask)

        if self.use_memory:
            word_embedding_c = word_embedding_c.view(batch_size, dialog_length, -1)
            memory_output = self.memory(word_embedding_c, sequence_output, party_mask)
            sequence_output = torch.cat((sequence_output, memory_output), 2)
        sequence_output = self.dropout(sequence_output)
        #
        if self.use_party:
            party_embeddings = self.party_embedding(party_ids)
            sequence_output = torch.cat((sequence_output, party_embeddings), dim=2)
        #
        if self.utterance_encoder is not None:
            self.utterance_encoder.flatten_parameters()
            utterance_output, _ = self.utterance_encoder(sequence_output)
        else:
            utterance_output = sequence_output

        if self.use_department:
            department_embeddings = self.department_embedding(department_ids)
            utterance_output = torch.cat((utterance_output, department_embeddings), dim=2)
        if self.use_disease:
            disease_embeddings = self.disease_embedding(disease_ids)
            utterance_output = torch.cat((utterance_output, disease_embeddings), dim=2)

        tmp_label_mask = torch.stack([label_mask] * utterance_output.shape[-1], 2)
        utterance_output = torch.mul(utterance_output, tmp_label_mask)
        logits = self.classifier(utterance_output)

        if labels is not None:
            if self.decoder == 'softmax':
                total_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.decoder == 'crf':
                total_loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask_label)
            else:
                raise ValueError()
            return total_loss
        else:
            if self.decoder == 'softmax':
                scores = torch.argmax(nn.functional.log_softmax(logits, dim=2), dim=2)
            elif self.decoder == 'crf':
                scores = self.crf.decode(logits, attention_mask_label)[0]
            else:
                raise ValueError()
            return scores

    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['max_ngram_size'] = args.max_ngram_size
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['use_memory'] = args.use_memory
        hyper_parameters['use_party'] = args.use_party
        hyper_parameters['use_department'] = args.use_department
        hyper_parameters['use_disease'] = args.use_disease
        hyper_parameters['utterance_encoder'] = args.utterance_encoder
        hyper_parameters['decoder'] = args.decoder
        hyper_parameters['lstm_hidden_size'] = args.lstm_hidden_size
        hyper_parameters['max_dialog_length'] = args.max_dialog_length
        return hyper_parameters

    @classmethod
    def load_model(cls, model_path):
        label2id = load_json(os.path.join(model_path, 'label2id.json'))
        hpara = load_json(os.path.join(model_path, 'hpara.json'))

        department2id_path = os.path.join(model_path, 'department2id.json')
        department2id = load_json(department2id_path) if os.path.exists(department2id_path) else None

        word2id_path = os.path.join(model_path, 'word2id.json')
        word2id = load_json(word2id_path) if os.path.exists(word2id_path) else None

        disease2id_path = os.path.join(model_path, 'disease2id.json')
        disease2id = load_json(disease2id_path) if os.path.exists(disease2id_path) else None

        res = cls(model_path=model_path, label2id=label2id, hpara=hpara,
                  department2id=department2id, word2id=word2id, disease2id=disease2id)

        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
        return res

    def save_model(self, output_dir, vocab_dir):
        output_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        label_map_file = os.path.join(output_dir, 'label2id.json')

        if not os.path.exists(label_map_file):
            save_json(label_map_file, self.label2id)

            save_json(os.path.join(output_dir, 'hpara.json'), self.hpara)
            if self.department2id is not None:
                save_json(os.path.join(output_dir, 'department2id.json'), self.department2id)
            if self.word2id is not None:
                save_json(os.path.join(output_dir, 'word2id.json'), self.word2id)
            if self.disease2id is not None:
                save_json(os.path.join(output_dir, 'disease2id.json'), self.disease2id)

            output_config_file = os.path.join(output_dir, 'config.json')
            with open(output_config_file, "w", encoding='utf-8') as writer:
                if self.bert:
                    writer.write(self.bert.config.to_json_string())
                elif self.zen:
                    writer.write(self.zen.config.to_json_string())
                else:
                    raise ValueError()
            output_bert_config_file = os.path.join(output_dir, 'bert_config.json')
            command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
            subprocess.run(command, shell=True)

            if self.bert or self.zen:
                vocab_name = 'vocab.txt'
            else:
                raise ValueError()
            vocab_path = os.path.join(vocab_dir, vocab_name)
            command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(output_dir, vocab_name))
            subprocess.run(command, shell=True)

            if self.zen:
                ngram_name = 'ngram.txt'
                ngram_path = os.path.join(vocab_dir, ngram_name)
                command = 'cp ' + str(ngram_path) + ' ' + str(os.path.join(output_dir, ngram_name))
                subprocess.run(command, shell=True)

    @staticmethod
    def data2example(data, flag=''):
        examples = []
        for i, (utterance, label, party, summary, max_utterance_len, party_mask, department, disease) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            text_a = utterance
            text_b = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                         party=party, summary=summary, max_utterance_len=max_utterance_len,
                                         party_mask=party_mask, department=department, disease=disease))
        return examples

    def convert_examples_to_features(self, examples):

        features = []

        tokenizer = self.zen_tokenizer if self.zen_tokenizer is not None else self.bert_tokenizer

        # -------- max ngram size --------
        max_utterance_length = min(int(max([example.max_utterance_len for example in examples]) * 1.1 + 2),
                                   self.max_seq_length)
        max_seq_length = max_utterance_length
        max_dialog_length = min(max(max([len(example.text_a) for example in examples]), 1), self.max_dialog_length)
        # -------- max ngram size --------

        for (ex_index, example) in enumerate(examples):
            valid = [[] for _ in range(max_dialog_length)]
            tokens = [[] for _ in range(max_dialog_length)]
            segment_ids = [[] for _ in range(max_dialog_length)]
            input_ids = [[] for _ in range(max_dialog_length)]
            input_mask = [[] for _ in range(max_dialog_length)]
            input_id_len = [1 for _ in range(max_dialog_length)]
            party_mask = [[] for _ in range(max_dialog_length)]

            for i in range(max_dialog_length):
                if i < len(example.text_a):
                    utterance = example.text_a[i]
                    party = example.party[i]
                    if party == 'P':
                        party_mask[i] = example.party_mask['P']
                    elif party == 'D':
                        party_mask[i] = example.party_mask['D']
                    else:
                        raise ValueError()
                    if len(party_mask[i]) > max_dialog_length:
                        party_mask[i] = party_mask[i][:max_dialog_length]
                    while len(party_mask[i]) < max_dialog_length:
                        party_mask[i].append(0)
                    for word in utterance:
                        token = tokenizer.tokenize(word)
                        tokens[i].extend(token)
                        for m in range(len(token)):
                            if m == 0:
                                valid[i].append(1)
                            else:
                                valid[i].append(0)
                    if len(tokens[i]) >= max_utterance_length - 1:
                        tokens[i] = tokens[i][0:(max_utterance_length - 2)]
                        valid[i] = valid[i][0:(max_utterance_length - 2)]

                    ntokens = []

                    ntokens.append("[CLS]")
                    segment_ids[i].append(0)

                    valid[i].insert(0, 1)

                    for token in tokens[i]:
                        ntokens.append(token)
                        segment_ids[i].append(0)
                    ntokens.append("[SEP]")

                    segment_ids[i].append(0)
                    valid[i].append(1)

                    # ntokens: ['[CLS]', '我' ... , '人', '[SEP]'] length: 5 + 2
                    # valid: [1, ..., 1] length 5 + 2 (前后加 1)
                    # label_mask: [1, ..., 1] length 5 + 2 (前后加 1)
                    # label_ids: [6, 5, 5, 2, 3, 4, 7] (前后加 [CLS] 和 [SEP] 的标签) length 5 + 2
                    # segment_id: [0, 0, ..., 0] length 7

                    input_ids[i] = tokenizer.convert_tokens_to_ids(ntokens)
                    # input_ids: [1, 2, 3, .. , 7] length 7
                    for _ in range(len(input_ids[i])):
                        input_mask[i].append(1)

                input_id_len[i] = len(input_ids[i])
                while len(input_ids[i]) < max_utterance_length:
                    input_ids[i].append(0)
                    input_mask[i].append(0)
                    segment_ids[i].append(0)
                    valid[i].append(1)
                while len(party_mask[i]) < max_dialog_length:
                    party_mask[i].append(0)
                assert len(input_ids[i]) == len(input_mask[i])
                assert len(input_ids[i]) == len(segment_ids[i])
                assert len(input_ids[i]) == len(valid[i])

            assert len(input_ids) == max_dialog_length
            assert len(input_ids[-1]) == max_utterance_length

            labellist = example.label
            label_mask = []
            label_ids = []

            for label in labellist:
                label_id = self.label2id[label] if label in self.label2id else self.label2id['<UNK>']
                label_ids.append(label_id)
                label_mask.append(1)
            if len(label_ids) > max_dialog_length:
                label_ids = label_ids[:max_dialog_length]
                label_mask = label_mask[:max_dialog_length]
            while len(label_ids) < max_dialog_length:
                label_ids.append(0)
                label_mask.append(0)

            partylist = example.party

            if self.party2id is not None:
                party_ids = []
                for party in partylist:
                    party_ids.append(self.party2id[party])
                if len(party_ids) > max_dialog_length:
                    party_ids = party_ids[:max_dialog_length]
                while len(party_ids) < max_dialog_length:
                    party_ids.append(0)
            else:
                party_ids = None

            if self.department2id is not None:
                department_ids = []
                if example.department in self.department2id:
                    department_id = self.department2id[example.department]
                else:
                    department_id = self.department2id['<UNK>']

                for _ in partylist:
                    department_ids.append(department_id)

                if len(department_ids) > max_dialog_length:
                    department_ids = department_ids[:max_dialog_length]

                while len(department_ids) < max_dialog_length:
                    department_ids.append(0)
            else:
                department_ids = None

            if self.disease2id is not None:
                disease_ids = []
                if example.disease in self.disease2id:
                    disease_id = self.disease2id[example.disease]
                else:
                    disease_id = self.disease2id['<UNK>']

                for _ in partylist:
                    disease_ids.append(disease_id)

                if len(disease_ids) > max_dialog_length:
                    disease_ids = disease_ids[:max_dialog_length]
                while len(disease_ids) < max_dialog_length:
                    disease_ids.append(0)
            else:
                disease_ids = None

            assert len(label_ids) == len(label_mask)
            assert len(label_ids) == max_dialog_length
            assert len(label_ids) == len(party_mask)
            assert len(label_ids) == len(party_mask[-1])

            if self.zen_ngram_dict is not None:
                all_ngram_ids = []
                all_ngram_positions_matrix = []
                # all_ngram_lengths = []
                # all_ngram_tuples = []
                # all_ngram_seg_ids = []
                # all_ngram_mask_array = []

                for token_list in tokens:
                    ngram_matches = []
                    #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                    for p in range(2, 8):
                        for q in range(0, len(token_list) - p + 1):
                            character_segment = token_list[q:q + p]
                            # j is the starting position of the ngram
                            # i is the length of the current ngram
                            character_segment = tuple(character_segment)
                            if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                                ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                                ngram_matches.append([ngram_index, q, p, character_segment])

                    # random.shuffle(ngram_matches)
                    ngram_matches = sorted(ngram_matches, key=lambda s: s[0])

                    max_ngram_in_seq_proportion = math.ceil(
                        (len(token_list) / max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                    if len(ngram_matches) > max_ngram_in_seq_proportion:
                        ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                    ngram_ids = [ngram[0] for ngram in ngram_matches]
                    ngram_positions = [ngram[1] for ngram in ngram_matches]
                    ngram_lengths = [ngram[2] for ngram in ngram_matches]
                    # ngram_tuples = [ngram[3] for ngram in ngram_matches]
                    # ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                    ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                    ngram_mask_array[:len(ngram_ids)] = 1

                    # record the masked positions
                    ngram_positions_matrix = np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq),
                                                      dtype=np.int32)
                    for i in range(len(ngram_ids)):
                        ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                    # Zero-pad up to the max ngram in seq length.
                    padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                    ngram_ids += padding
                    # ngram_lengths += padding
                    # ngram_seg_ids += padding

                    all_ngram_ids.append(ngram_ids)
                    all_ngram_positions_matrix.append(ngram_positions_matrix)
                    # all_ngram_lengths.append(ngram_lengths)
                    # all_ngram_tuples.append(ngram_tuples)
                    # all_ngram_seg_ids.append(ngram_seg_ids)
                    # all_ngram_mask_array.append(ngram_mask_array)
                while len(all_ngram_ids) < max_dialog_length:
                    all_ngram_ids.append([0] * self.zen_ngram_dict.max_ngram_in_seq)
                    all_ngram_positions_matrix.append(np.zeros(shape=(max_seq_length, self.zen_ngram_dict.max_ngram_in_seq),
                                                      dtype=np.int32))
            else:
                all_ngram_ids = None
                all_ngram_positions_matrix = None
                # all_ngram_lengths = None
                # all_ngram_tuples = None
                # all_ngram_seg_ids = None
                # all_ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              input_id_len=input_id_len,
                              party_mask=party_mask,
                              party=party_ids,
                              department=department_ids,
                              disease=disease_ids,
                              ngram_ids=all_ngram_ids,
                              ngram_positions=all_ngram_positions_matrix,
                              # ngram_lengths=all_ngram_lengths,
                              # ngram_tuples=all_ngram_tuples,
                              # ngram_seg_ids=all_ngram_seg_ids,
                              # ngram_masks=all_ngram_mask_array
                              ))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.long)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)

        all_lmask = torch.tensor([f.label_mask for f in feature], dtype=torch.float)
        lmask = all_lmask.to(device)

        if self.memory is not None:
            all_party_mask = torch.tensor([f.party_mask for f in feature], dtype=torch.float)

            party_mask = all_party_mask.to(device)
        else:
            party_mask = None

        if self.use_party:
            all_party_ids = torch.tensor([f.party for f in feature], dtype=torch.long)
            party_ids = all_party_ids.to(device)
        else:
            party_ids = None

        if self.use_department:
            all_department_ids = torch.tensor([f.department for f in feature], dtype=torch.long)
            department_ids = all_department_ids.to(device)
        else:
            department_ids = None

        if self.use_disease:
            all_disease_ids = torch.tensor([f.disease for f in feature], dtype=torch.long)
            disease_ids = all_disease_ids.to(device)
        else:
            disease_ids = None

        if self.zen is not None:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
            # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
            # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
            # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None
        return input_ids, input_mask, l_mask, label_ids, ngram_ids, ngram_positions, segment_ids, valid_ids, \
               lmask, party_mask, party_ids, department_ids, disease_ids


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, party=None, summary=None, max_utterance_len=None,
                 party_mask=None, department=None, disease=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.party = party
        self.summary = summary
        self.max_utterance_len = max_utterance_len
        self.party_mask = party_mask
        self.department = department
        self.disease = disease


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None,
                 input_id_len=None, party_mask=None, party=None, department=None, disease=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.input_id_len = input_id_len
        self.party_mask = party_mask
        self.party = party
        self.department = department
        self.disease = disease

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def readsentence(filename):
    data = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            label_list = ['S' for _ in range(len(line))]
            data.append((line, label_list))
    return data

