# Data Pre-processing

## Requirement

`lxml` is required to obtain the data.

## Obtain the data

Run `./crawl_data.sh` to obtain the full data. The code will crawl the data in the `data/urls/url_list.txt`. The dialogs will stored under `data/crawl_data/data` and the html files will stored under `data/crawl_data/html`.

If the code crash in the middle, you just need to run `./crawl_data.sh` again. The code will start obtaining the data from where it stops last time.

Usage:
* `--chunk_size`: the number of dialogs to store in each file.

**Note**: we tested our script recently and found that there are problems in the online sources (e.g., the urls/dialog/summaries do not exist). Therefore, it is possible that you cannot obtain the full resource. If the problem occurs, the code will save the problem to `data/crawl_data/warnings` and continue downloading the rest data. You can submit an issue with the warning file so that we can try to address the problems.

## Merge the data with our silver standard.

In processing ...

You can find the train/test split under [`data/splits`](data/splits). You can find our annotations of each utterance ("1" stands for problem description; "2" stands for doctor suggestions; "0" stands for other utterances) under [`data/annotations`](data/annotations).
