import requests
import os
import json
from lxml import html
import argparse
import datetime
import time
import random
from selenium import webdriver

SUMMARY_RULE = [
    '//main[@class="content"]/section/div[@class="qa-arrangement"]/div[@class="qa-arrangement-body"]/div[@class="qa-title"]',
    '//main/div[@class="qa-arrangement-body"]/p'
]


CONTENT_RULE = [
    '//main[@class="content"]/section/section[@class="problem-detail-wrap"]/section[@class="problem-detail-inner"]/div[@class="block-line"]/div[@class="block-right"]',
    '//main/dev[@class="problem-detail-wrap"]/div[@class="block-line"]'
]

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
    existing_ids = set()
    for file in sorted(all_files, reverse=True):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(data_dir, file), 'r', encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            dialog = json.loads(line)
            existing_ids.add(dialog['id'])
    return existing_ids


def strip_str(string):
    if string is None:
        return ''
    return ','.join(string.strip().split())


def name_filter(string):
    if not string == '患者':
        return '医生'
    else:
        return '患者'


def crawl_dialog(dialog, sleep_time, driver):
    dialog_id = dialog['id']
    dialog_url = dialog['url']
    exception_report = []

    dialog_dict = {'id': dialog_id, 'url': dialog_url, 'content': [], 'summary': {'description': '', 'suggestion': ''}}
    html_dict = {'id': dialog_id, 'url': dialog_url, 'html': ''}

    time.sleep(1)
    response = requests.get(dialog_url)

    step_threshold = 5
    try_times = 0
    while (response.status_code == 503 or response.status_code == 429) and try_times < 20:
        time.sleep(max(sleep_time + 4 * (random.random() - 0.5), 2))
        response = requests.get(dialog_url)
        try_times += 1
        sleep_time += int(try_times/step_threshold)

    if response.status_code == 200:
        response_text = response.text
        html_dict['html'] = response_text
        html_format = html.fromstring(response_text)

        # get dialog
        speaker_info = []
        for context_rule in CONTENT_RULE:
            speaker_rule = context_rule + '/h6'
            utterance_rule = context_rule + '/p'
            speaker_info = html_format.xpath(speaker_rule)
            utterances_info = html_format.xpath(utterance_rule)
            if len(speaker_info) > 0:
                speaker_list = [name_filter(sp.text.strip()) for sp in speaker_info]
                utterances_list = [strip_str(ut.text) for ut in utterances_info]

                for sp, ut in zip(speaker_list, utterances_list):
                    if not ut == '':
                        dialog_dict['content'].append(
                            {'speaker': sp,
                             'utterance': ut}
                        )
                break
        if len(speaker_info) == 0:
            warning_info = '%s %s dialog not found!' % (dialog_id, dialog_url)
            exception_report.append(warning_info)
            print(warning_info)

        # get the summary
        for sum_rule in SUMMARY_RULE:
            summary_info = html_format.xpath(sum_rule)
            if len(summary_info) > 0 and summary_info[0].text is not None:
                description = summary_info[0].text
                if description.startswith('问题描述：'):
                    description = description[5:]

                dialog_dict['summary']['description'] = strip_str(description)

                suggestion = summary_info[1].text
                if suggestion.startswith('分析及建议：'):
                    suggestion = suggestion[6:]
                dialog_dict['summary']['suggestion'] = strip_str(suggestion)
                break

        if dialog_dict['summary']['suggestion'] == '':
            driver.implicitly_wait(sleep_time)
            driver.get(dialog_url)
            try:
                enter = driver.find_element_by_class_name('qa-arrangement-btn')
                driver.execute_script("arguments[0].scrollIntoView();", enter)
                if driver.find_element_by_class_name('qa-arrangement-body').is_displayed():
                    pass
                else:
                    enter.click()
                # text_result = driver.find_element_by_class_name('qa-arrangement-body').text
                text_result = driver.find_elements_by_class_name('qa-des')

                if len(text_result) == 2:
                    dialog_dict['summary']['description'] = text_result[0].text
                    dialog_dict['summary']['suggestion'] = text_result[1].text
            except Exception:
                pass

        if dialog_dict['summary']['suggestion'] == '':
            if len(dialog_dict['content']) > 0:
                description = dialog_dict['content'][0]['utterance']
                dialog_dict['summary']['description'] = description
                new_dialog_content = []
                for utt in dialog_dict['content']:
                    speaker = utt['speaker']
                    if speaker == '医生' and utt['utterance'].startswith('针对本次问诊，医生更新了总结建议'):
                        dialog_dict['summary']['suggestion'] += utt['utterance']
                    else:
                        new_dialog_content.append(
                            {'speaker': speaker,
                            'utterance': utt['utterance']}
                        )
                dialog_dict['content'] = new_dialog_content

            if dialog_dict['summary']['suggestion'] == '':
                warning_info = '%s %s summary not found!' % (dialog_id, dialog_url)
                exception_report.append(warning_info)
                print(warning_info)
    else:
        warning_info = '%s %d %s URL not found!' % (dialog_id, response.status_code, dialog_url)
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
    parser.add_argument("--sleep_time", default=15, type=int)
    parser.add_argument("--test_url", default=None, type=str)
    parser.add_argument("--chrome_driver", default='./chromedriver', type=str)

    args = parser.parse_args()

    full_dialogs = get_full_dialog_list(FULL_URL_FILE)
    existing_ids = get_existing_index(JSON_DATA_DIR)
    existing_dialog_num = len(existing_ids)

    all_dialogs = []

    for dialog in full_dialogs:
        if not dialog['id'] in existing_ids:
            all_dialogs.append(dialog)

    print('%d dialogs exists in %s' % (existing_dialog_num, JSON_DATA_DIR))
    print('%d dialogs to be crawl' % (len(all_dialogs)))

    chunk_dialogs = []
    chunk_htmls = []

    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    warning_file = 'warning-%s' % now_time
    warning_file = os.path.join(WARNINGS_DIR, warning_file)

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    driver = webdriver.Chrome(executable_path=args.chrome_driver, options=chrome_options)

    # test code
    if args.test_url is not None:
        test_dialog = {'id': 00000, 'url': args.test_url}
        dialog_dict, html_dict, exception_report = crawl_dialog(test_dialog, sleep_time=args.sleep_time, driver=driver)
        exit(0)

    for i in range(len(all_dialogs)):

        current_index = existing_dialog_num + i + 1

        print('Processing %05d / %d' % (current_index, len(full_dialogs)))

        dialog = all_dialogs[i]

        dialog_dict, html_dict, exception_report = crawl_dialog(dialog, sleep_time=args.sleep_time, driver=driver)

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

