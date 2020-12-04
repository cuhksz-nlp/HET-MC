import json


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        line = f.readline()
    return json.loads(line)


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f)
        f.write('\n')


def read_dialog(filename):
    dialog_data = []
    utterance = []
    max_utterance_len = -1
    label = []
    party = []
    party_mask = {'P': [], 'D': []}
    disease = ''
    department = ''
    # summary = {'SUM1_ORIG': None, 'SUM1': None, 'SUM2_ORIG': None, 'SUM2A': None, 'SUM2B': None}
    summary = {'SUM1': '', 'SUM2': '', 'SUM2A': '', 'SUM2B': ''}
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                if len(utterance) > 0:
                    dialog_data.append((utterance, label, party, summary, max_utterance_len, party_mask,
                                        department, disease))
                    utterance = []
                    max_utterance_len = -1
                    label = []
                    party = []
                    disease = ''
                    department = ''
                    # summary = {'SUM1_ORIG': None, 'SUM1': None, 'SUM2_ORIG': None, 'SUM2A': None, 'SUM2B': None}
                    summary = {'SUM1': '', 'SUM2': '', 'SUM2A': '', 'SUM2B': ''}
                    party_mask = {'P': [], 'D': []}
            elif line.startswith('P') or line.startswith('D'):
                splits = line.split('\t')
                party.append(splits[0])
                utterance.append(splits[1])
                label.append(splits[2])
                if party[-1] == 'P':
                    party_mask['P'].append(1)
                    party_mask['D'].append(0)
                else:
                    party_mask['P'].append(0)
                    party_mask['D'].append(1)
                if len(utterance) > max_utterance_len:
                    max_utterance_len = len(utterance)
            elif line.startswith('id'):
                splits = line.split('\t')
                department = splits[3]
                disease = splits[5]
                continue
            elif line.startswith('SUM1'):
                splits = line.split('\t')
                summary[splits[0]] += splits[1]
            elif line.startswith('SUM2'):
                splits = line.split('\t')
                summary['SUM2'] += splits[1]
                if line.startswith('SUM2.0'):
                    splits = line.split('\t')
                    summary['SUM2A'] += splits[1]
                elif line.startswith('SUM2.1'):
                    splits = line.split('\t')
                    summary['SUM2B'] += splits[1]

    if len(utterance) > 0:
        dialog_data.append((utterance, label, party, summary, max_utterance_len, party_mask, department, disease))
    return dialog_data


def get_vocab(training_dialog):
    label2id = {'<PAD>': 0, '<UNK>': 1}
    word2id = {'<PAD>': 0, '<UNK>': 1}
    department2id = {'<PAD>': 0, '<UNK>': 1}
    disease2id = {'<PAD>': 0, '<UNK>': 1}
    for dialog in training_dialog:
        utterance = dialog[0]
        label = dialog[1]
        department = dialog[6]
        disease = dialog[7]

        for w in utterance:
            if w not in word2id:
                word2id[w] = len(word2id)
        for l in label:
            if l not in label2id:
                label2id[l] = len(label2id)
        if department not in department2id:
            department2id[department] = len(department2id)
        if disease not in disease2id:
            disease2id[disease] = len(disease2id)

    return label2id, word2id, department2id, disease2id
