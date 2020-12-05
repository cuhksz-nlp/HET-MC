import requests
import os
import json
from lxml import html
import argparse
import datetime

FULL_URL_FILE = 'data/urls/url_list.txt'
CRAWL_DATA_DIR = 'data/crawl_data'
HTML_DATA_DIR = os.path.join(CRAWL_DATA_DIR, 'html')
JSON_DATA_DIR = os.path.join(CRAWL_DATA_DIR, 'data')
WARNINGS_DIR = os.path.join(CRAWL_DATA_DIR, 'warnings')

if not os.path.exists(JSON_DATA_DIR):
    os.makedirs(JSON_DATA_DIR)
if not os.path.exists(HTML_DATA_DIR):
    os.makedirs(HTML_DATA_DIR)
if not os.path.exists(WARNINGS_DIR):
    os.makedirs(WARNINGS_DIR)


def get_full_dialog_list(full_list_path):
    dialog_list = []
    with open(full_list_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        splits = line.split()
        dialog = {'id': splits[0], 'url': splits[1]}
        dialog_list.append(dialog)
    return dialog_list


def get_existing_index(data_dir):
    all_files = os.listdir(data_dir)
    for file in sorted(all_files, reverse=True):
        if not file.endswith('.json'):
            continue
        index_range = file[file.find('.')+1: file.rfind('.')]
        return int(index_range.split('_')[1])
    return 0


def strip_str(string):
    return ','.join(string.strip().split())


def name_filter(string):
    if not string == '患者':
        return '医生'
    else:
        return '患者'


def crawl_dialog(dialog):
    dialog_id = dialog['id']
    dialog_url = dialog['url']
    exception_report = []

    dialog_dict = {'id': dialog_id, 'url': dialog_url, 'content': [], 'summary': {'description': '', 'suggestion': ''}}
    html_dict = {'id': dialog_id, 'url': dialog_url, 'html': ''}

    try:
        response = requests.get(dialog_url)
    except Exception:
        warning_info = '%s URL not found!' % dialog_id
        exception_report.append(warning_info)
        print(warning_info)
    else:
        response_text = response.text
        html_dict['html'] = response_text
        html_format = html.fromstring(response_text)

        # get the summary
        summary_info = html_format.xpath('//main[@class="content"]'
                                         '/section/div[@class="qa-arrangement"]'
                                         '/div[@class="qa-arrangement-body"]'
                                         '/div[@class="qa-title"]')
        if len(summary_info) > 0:
            description = summary_info[0].text
            if description.startswith('问题描述：'):
                description = description[5:]

            dialog_dict['summary']['description'] = strip_str(description)

            suggestion = summary_info[1].text
            if suggestion.startswith('分析及建议：'):
                suggestion = suggestion[6:]
            dialog_dict['summary']['suggestion'] = strip_str(suggestion)
        else:
            warning_info = '%s summary not found!' % dialog_id
            exception_report.append(warning_info)
            print(warning_info)

        # get dialog
        speaker_info = html_format.xpath('//main[@class="content"]'
                                         '/section/section[@class="problem-detail-wrap"]'
                                         '/section[@class="problem-detail-inner"]'
                                         '/div[@class="block-line"]'
                                         '/div[@class="block-right"]'
                                         '/h6')
        utterances_info = html_format.xpath('//main[@class="content"]'
                                            '/section/section[@class="problem-detail-wrap"]'
                                            '/section[@class="problem-detail-inner"]'
                                            '/div[@class="block-line"]'
                                            '/div[@class="block-right"]'
                                            '/p')

        if len(speaker_info) > 0:
            speaker_list = [name_filter(sp.text.strip()) for sp in speaker_info]
            utterances_list = [strip_str(ut.text) for ut in utterances_info]

            for sp, ut in zip(speaker_list, utterances_list):
                if not ut == '':
                    dialog_dict['content'].append(
                        {'speaker': sp,
                         'utterance': ut}
                    )
        else:
            warning_info = '%s dialog not found!' % dialog_id
            exception_report.append(warning_info)
            print(warning_info)

    return dialog_dict, html_dict, exception_report


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def save_warning_file(warning_file, warnings):
    with open(warning_file, 'a', encoding='utf8') as f:
        for line in warnings:
            f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunk_size", default=500, type=int)

    args = parser.parse_args()

    full_dialogs = get_full_dialog_list(FULL_URL_FILE)
    existing_index = get_existing_index(JSON_DATA_DIR)

    all_dialogs = full_dialogs[existing_index:]

    print('%d dialogs exists in %s' % (existing_index, JSON_DATA_DIR))
    print('%d dialogs to be crawl' % (len(all_dialogs)))

    chunk_dialogs = []
    chunk_htmls = []

    all_index = all_dialogs[-1]['id']

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    warning_file = 'warning-%s' % now_time
    warning_file = os.path.join(WARNINGS_DIR, warning_file)

    for i in range(len(all_dialogs)):

        current_index = existing_index + i + 1

        print('Processing %05d / %s' % (current_index, all_index))

        dialog = all_dialogs[i]
        dialog_dict, html_dict, exception_report = crawl_dialog(dialog)

        if len(exception_report) > 0:
            save_warning_file(warning_file, exception_report)

        chunk_dialogs.append(dialog_dict)
        chunk_htmls.append(html_dict)

        if len(chunk_dialogs) == args.chunk_size:
            start_index = current_index - args.chunk_size + 1

            print('Saving ids from %05d to %05d' % (start_index, current_index))

            data_path = os.path.join(JSON_DATA_DIR, 'data.%05d_%05d.json' % (start_index, current_index))
            html_path = os.path.join(HTML_DATA_DIR, 'html.%05d_%05d.json' % (start_index, current_index))

            save_json(chunk_dialogs, data_path)
            save_json(chunk_htmls, html_path)

            chunk_dialogs = []
            chunk_htmls = []


if __name__ == "__main__":
    main()

