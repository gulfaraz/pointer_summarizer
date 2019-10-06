# Implementation of "Get To The Point: Summarization with Pointer-Generator Networks" ([https://arxiv.org/abs/1704.04368](https://arxiv.org/abs/1704.04368)) with additional Glove and Elmo embeddings

## Project for the "Deep Learning for Natural Language Processing" course at the University of Amsterdam

# Requirements

## Environment

* Python 3.6
* Install the requirements specified in requirements.txt

E.g.: run the following commands

```bash
virtualenv -p python3.6 .env
source .env/bin/activate
pip install -r requirements.txt
```

## Getting the data


Download the preprocessed cnn-dailymail dataset

E.g.

```bash
pip install gdown
gdown https://drive.google.com/uc?id=0BzQ6rtO2VN95a0c3TlZCWkl3aU0
unzip finished_files.zip
```

Or you can follow the instructions given at [https://github.com/abisee/cnn-dailymail](https://github.com/abisee/cnn-dailymail)


## PyRouge

To be able to generate the Rouge scores, you also need to setup pyrouge: [https://github.com/andersjo/pyrouge](https://github.com/andersjo/pyrouge)


## Config

Modify corresponding parts of the [config.py](data_util/config.py) file according to the location of the downloaded data. 


# How to run

## Training
Change the current directory to the [training\_ptr\_gen](training_ptr_gen) directory.

Run the [train.py](training_ptr_gen/train.py) script.

```
usage: train.py [-h] -m MODEL_FILE_PATH [-g] [-e]

Train script

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_FILE_PATH    Model file for retraining (default: None).
  -g, --finetune_glove  Finetune the glove embeddings
  -e, --use_elmo        Use elmo embeddings too during the training (no
                        finetuning).

```

## Generating the summaries

Run the [decode.py](training_ptr_gen/decode.py) script.

```
usage: decode.py [-h] -m MODEL_FILENAME [-e]

Decode script

optional arguments:
  -h, --help         show this help message and exit
  -m MODEL_FILENAME  Saved model file from training. This will be used to get
                     the summaries
  -e, --use_elmo     Use elmo embeddings too (must match the model), or glove
                     only

```

## Evaluation

Run the [training\_ptr\_gen/eval.py](training_ptr_gen/eval.py) script.

```
usage: eval.py [-h] -m MODEL_FILENAME

Eval script

optional arguments:
  -h, --help         show this help message and exit
  -m MODEL_FILENAME  Saved model file from training.

```

## Bayesian Dropout experiments

Run the [bayesion\_dropout.py](bayesian_dropout.py) script.

```

usage: bayesian_dropout.py [-h] -m MODEL -o OUTPUT_DIR [-n NUM_EXPERIMENTS]
                           [-s MAX_NUM_SUMMARIES] [-b BEGINNING]
                           [-l MAX_SENTENCE_LENGTH] [-d] [-e]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model file path
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output path for saved probabilities
  -n NUM_EXPERIMENTS, --num_experiments NUM_EXPERIMENTS
                        How many different outputs we would like to get for
                        the same input
  -s MAX_NUM_SUMMARIES, --max_num_summaries MAX_NUM_SUMMARIES
                        Run the bayesian dropout on this many examples only
  -b BEGINNING, --beginning BEGINNING
                        Begin with this summary, not the first one.
  -l MAX_SENTENCE_LENGTH, --max_sentence_length MAX_SENTENCE_LENGTH
                        Only for testing
  -d, --dont_use_gpu    This flag will try to disable GPU usage
  -e, --use_elmo        Use Glove+Elmo embeddings together, otherwise only
                        Glove

```


> # Original readme

> pytorch implementation of *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*

> Train with pointer generation + coverage loss enabled 
> --------------------------------------------
> After training for 100k iterations with coverage loss enabled (batch size 8)

> ```
> ROUGE-1:
> rouge_1_f_score: 0.3907 with confidence interval (0.3885, 0.3928)
> rouge_1_recall: 0.4434 with confidence interval (0.4410, 0.4460)
> rouge_1_precision: 0.3698 with confidence interval (0.3672, 0.3721)

> ROUGE-2:
> rouge_2_f_score: 0.1697 with confidence interval (0.1674, 0.1720)
> rouge_2_recall: 0.1920 with confidence interval (0.1894, 0.1945)
> rouge_2_precision: 0.1614 with confidence interval (0.1590, 0.1636)

> ROUGE-l:
> rouge_l_f_score: 0.3587 with confidence interval (0.3565, 0.3608)
> rouge_l_recall: 0.4067 with confidence interval (0.4042, 0.4092)
> rouge_l_precision: 0.3397 with confidence interval (0.3371, 0.3420)
> ```

> ![Alt text](learning_curve_coverage.png?raw=true "Learning Curve with coverage loss")

> Training with pointer generation enabled
> --------------------------------------------

> After training for 500k iterations (batch size 8)

> ```
> ROUGE-1:
> rouge_1_f_score: 0.3500 with confidence interval (0.3477, 0.3523)
> rouge_1_recall: 0.3718 with confidence interval (0.3693, 0.3745)
> rouge_1_precision: 0.3529 with confidence interval (0.3501, 0.3555)

> ROUGE-2:
> rouge_2_f_score: 0.1486 with confidence interval (0.1465, 0.1508)
> rouge_2_recall: 0.1573 with confidence interval (0.1551, 0.1597)
> rouge_2_precision: 0.1506 with confidence interval (0.1483, 0.1529)

> ROUGE-l:
> rouge_l_f_score: 0.3202 with confidence interval (0.3179, 0.3225)
> rouge_l_recall: 0.3399 with confidence interval (0.3374, 0.3426)
> rouge_l_precision: 0.3231 with confidence interval (0.3205, 0.3256)
> ```
> ![Alt text](learning_curve.png?raw=true "Learning Curve with pointer generation")


> How to run training:
> --------------------------------------------
> 1) Follow data generation instruction from https://github.com/abisee/cnn-dailymail
> 2) Run start_train.sh, you might need to change some path and parameters in data_util/config.py
> 3) For training run start_train.sh, for decoding run start_decode.sh, and for evaluating run run_eval.sh

> Note:
> * It is tested on pytorch 0.4 with python 2.7
> * You need to setup [pyrouge](https://github.com/andersjo/pyrouge) to get the rouge score



