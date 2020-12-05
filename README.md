# HET-MC

This is the implementation of [Summarizing Medical Conversations via Identifying Important Utterances](https://www.aclweb.org/anthology/2020.coling-main.63/) at COLING 2020.

You can e-mail Yuanhe Tian at `yhtian@uw.edu` or Guimin Chen at `cuhksz.nlp@gmail.com`, if you have any questions.

## Citation

If you use or extend our work, please cite our paper at COLING 2020.

```
@inproceedings{song-etal-2020-summarizing,
    title = "Summarizing Medical Conversations via Identifying Important Utterances",
    author = "Song, Yan and Tian, Yuanhe and Wang, Nan and Xia, Fei",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    pages = "717--729",
}
```

## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.3`

## Dataset

To obtain the data, you can go to [`data_preprocessing`](./data_preprocessing) directory for details.

## Downloading BERT, ZEN and HET-MC

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) and ZEN ([paper](https://arxiv.org/abs/1911.00720)) as the encoder.

For BERT, please download pre-trained BERT-Base Chinese from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For ZEN, you can download the pre-trained model from [here](https://github.com/sinovation/ZEN).

For HET-MC, you can download the models we trained in our experiments from [here](https://pan.baidu.com/s/17peaEeqqu0ck96A2KyoDDg) (passcode: b1w1).

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

## Training and Testing

You can find the command lines to train and test models in `run.sh`.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_test`: test the model.
* `--use_bert`: use BERT as token encoder.
* `--use_zen`: use ZEN as token encoder.
* `--bert_model`: the directory of pre-trained BERT/ZEN model.
* `--use_memory`: use memories.
* `--utterance_encoder`: the utterance encoder to be used (should be one of `none`, `LSTM`, and `biLSTM`).
* `--lstm_hidden_size`: the size of hidden state in the LSTM/biLSTM utterance encoder.
* `--decoder`: the decoder to be used (can be either `crf` or `softmax`).
* `--use_party`: use the speaker role information.
* `--use_department`: use the department information.
* `--use_disease`: use disease information
* `--model_name`: the name of model to save.

## To-do List

* Release the code to get the data.
* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.

