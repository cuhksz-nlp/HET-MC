# Data Pre-processing

## Requirement

* Packages: `lxml` and `selenium`
* Google Chrome Browser and [Google Chrome Driver](https://sites.google.com/a/chromium.org/chromedriver/downloads) (the version of the driver should match your browser).

Following the following steps to install Google Chrome Browser and Google Chrome Driver if you are running the crawler on a Linux cloud server:
* Download and install the latest version of Google Chrome:  
```
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
```
* Download the [Google Chrome Driver]((https://sites.google.com/a/chromium.org/chromedriver/downloads)) and put it under the current directory. You can use `google-chrome --version` to find the version of Google Chrome Browser. You also need to add the execute permission to the driver by `chmod +x ./chromedriver`.

## Obtain the data

Run `./crawl_data.sh` to obtain the full data. The code will crawl the data in the `data/urls/url_list.txt`. The dialogs will stored under `data/crawl_data/data` and the html files will stored under `data/crawl_data/html`.

If the code crash in the middle, you just need to run `./crawl_data.sh` again. The code will start obtaining the data from where it stops last time.

Usage:
* `--chunk_size`: the number of dialogs to store in each file.
* `--sleep_time`: the time (in seconds) to wait between every two requests. Increase this number if necessary.
* `--chrome_driver`: the path to Google Chrome Driver.

**Note**: we tested our script and found that there are problems in the online sources (e.g., the urls/dialog/summaries do not exist). Therefore, it is possible that you cannot obtain the full resource. If the problem occurs, the code will save the problem to `data/crawl_data/warnings` and continue downloading the rest data. You can submit an issue or send us an e-mail with the warning file and we will try to address the problems.

## Updates

**July 3, 2021**: The old version of the crawler does not work because the platform updated their web pages. We update our crawler accordingly, which requires Google Chrome Browser and Google Chrome Driver.

## Merge the data with our silver standard.

You can find the train/test split under [`data/splits`](data/splits). You can find our annotations of each utterance ("1" stands for problem description; "2" stands for doctor suggestions; "0" stands for other utterances) under [`data/annotations`](data/annotations).
