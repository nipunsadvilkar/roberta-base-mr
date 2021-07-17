# RoBERTa base model for Marathi language
Pretrained model on Marathi language using a masked language modeling (MLM) objective. RoBERTa was introduced in
[this paper](https://arxiv.org/abs/1907.11692) and first released in
[this repository](https://github.com/pytorch/fairseq/tree/master/examples/roberta). We trained RoBERTa model for Marathi Language during community week hosted by Huggingface 🤗 using JAX/Flax for NLP & CV jax.
<h3 align="center">
  <p>RoBERTa base model for Marathi language</p>
  <img src="huggingface-marathi-roberta.png" alt="huggingface-marathi-roberta" width="350" height="350">
<h3 align="center">

## Model description
Marathi RoBERTa is a transformers model pretrained on a large corpus of Marathi data in a self-supervised fashion. 

## Intended uses & limitations
You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task.
Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. We used this model to fine tune on text classification task for iNLTK and indicNLP news text classification problem statement. Since marathi mc4 dataset is made by scraping marathi newspapers text, it will involve some biases which will also affect all fine-tuned versions of this model.

### How to use
You can use this model directly with a pipeline for masked language modeling:
```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='flax-community/roberta-base-mr')
>>> unmasker("मोठी बातमी! उद्या दुपारी <mask> वाजता जाहीर होणार दहावीचा निकाल")
[{'score': 0.057209037244319916,'sequence': 'मोठी बातमी! उद्या दुपारी आठ वाजता जाहीर होणार दहावीचा निकाल',
  'token': 2226,
  'token_str': 'आठ'},
 {'score': 0.02796074189245701,
  'sequence': 'मोठी बातमी! उद्या दुपारी २० वाजता जाहीर होणार दहावीचा निकाल',
  'token': 987,
  'token_str': '२०'},
 {'score': 0.017235398292541504,
  'sequence': 'मोठी बातमी! उद्या दुपारी नऊ वाजता जाहीर होणार दहावीचा निकाल',
  'token': 4080,
  'token_str': 'नऊ'},
 {'score': 0.01691395975649357,
  'sequence': 'मोठी बातमी! उद्या दुपारी २१ वाजता जाहीर होणार दहावीचा निकाल',
  'token': 1944,
  'token_str': '२१'},
 {'score': 0.016252165660262108,
  'sequence': 'मोठी बातमी! उद्या दुपारी  ३ वाजता जाहीर होणार दहावीचा निकाल',
  'token': 549,
  'token_str': ' ३'}]
```

## Training data
The RoBERTa Marathi model was pretrained on `mr` dataset of C4 multilingual dataset:
<br>
<br>
[C4 (Colossal Clean Crawled Corpus)](https://yknzhu.wixsite.com/mbweb), Introduced by Raffel et al. in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://paperswithcode.com/paper/exploring-the-limits-of-transfer-learning). 

The dataset can be downloaded in a pre-processed form from [allennlp](https://github.com/allenai/allennlp/discussions/5056) or huggingface's datsets - [mc4 dataset](https://huggingface.co/datasets/mc4).
  Marathi (`mr`) dataset consists of 14 billion tokens, 7.8 million docs and with weight ~70 GB of text.

## Data Cleaning

Though initial `mc4` marathi corpus size ~70 GB, Through data exploration, it was observed it contains docs from different languages especially thai, chinese etc. So we had to clean the dataset before traning tokenizer and model. Surprisingly, results after cleaning Marathi mc4 corpus data:

#### **Train set:**
Clean docs count 1581396 out of 7774331. <br>
**~20.34%** of whole marathi train split is actually Marathi.

#### **Validation set**
Clean docs count 1700 out of 7928. <br>
**~19.90%** of whole marathi validation split is actually Marathi.

## Training procedure
### Preprocessing
The texts are tokenized using a byte version of Byte-Pair Encoding (BPE) and a vocabulary size of 50265. The inputs of
the model take pieces of 512 contiguous token that may span over documents. The beginning of a new document is marked
with `<s>` and the end of one by `</s>`
The details of the masking procedure for each sentence are the following:
- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `<mask>`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.
Contrary to BERT, the masking is done dynamically during pretraining (e.g., it changes at each epoch and is not fixed).
### Pretraining
The model was trained on Google Cloud Engine TPUv3-8 machine (with 335 GB of RAM, 1000 GB of hard drive, 96 CPU cores) **8 v3 TPU cores** for 42K steps with a batch size of 128 and a sequence length of 128. The
optimizer used is Adam with a learning rate of 3e-4, \\(\beta_{1} = 0.9\\), \\(\beta_{2} = 0.98\\) and
\\(\epsilon = 1e-8\\), a weight decay of 0.01, learning rate warmup for 1,000 steps and linear decay of the learning
rate after.

We tracked experiments and hyperparameter tunning on weights and biases platform. Here is link to main dashboard: <br>
[Link to Weights and Biases Dashboard for Marathi RoBERTa model](https://wandb.ai/nipunsadvilkar/roberta-base-mr/runs/19qtskbg?workspace=user-nipunsadvilkar)

#### **Pretraining Results**

RoBERTa Model reached **eval accuracy of 85.28%** around ~35K step **with train loss at 0.6507 and eval loss at 0.6219**.


## Fine Tuning on downstream tasks
We performed fine-tuning on downstream tasks. We used following datasets for classification:

1. [IndicNLP Marathi news classification](https://github.com/ai4bharat-indicnlp/indicnlp_corpus#publicly-available-classification-datasets)
2. [iNLTK Marathi news headline classification](https://www.kaggle.com/disisbig/marathi-news-dataset)

### **Fine tuning on downstream task results (Segregated)**

#### 1. [IndicNLP Marathi news classification](https://github.com/ai4bharat-indicnlp/indicnlp_corpus#publicly-available-classification-datasets)

IndicNLP Marathi news dataset consists 3 classes - `['lifestyle', 'entertainment', 'sports']` - with following docs distribution as per classes:

| train | eval | test
| -- | -- | --
| 9672 | 477 | 478

Our Marathi RoBERTa `roberta-base-mr` model outperformed both classifier mentioned in [Arora, G. (2020). iNLTK](https://www.semanticscholar.org/paper/iNLTK%3A-Natural-Language-Toolkit-for-Indic-Languages-Arora/5039ed9e100d3a1cbbc25a02c82f6ee181609e83/figure/3) and [Kunchukuttan, Anoop et al. AI4Bharat-IndicNLP.](https://www.semanticscholar.org/paper/AI4Bharat-IndicNLP-Corpus%3A-Monolingual-Corpora-and-Kunchukuttan-Kakwani/7997d432925aff0ba05497d2893c09918298ca55/figure/4)


Dataset | FT-W | FT-WC | INLP | iNLTK | robera-base-mr
-- | -- | -- | -- | -- | --
iNLTK Headlines | 83.06 | 81.65 | 89.92 | 92.4 | **97.48**

**🤗 Huggingface Model hub repo:**<br>
`roberta-base-mr` fine tuned on iNLTK Headlines classification dataset model:

[**`flax-community/mr-indicnlp-classifier`**](https://huggingface.co/flax-community/mr-indicnlp-classifier)

Fine tuning experiment's weight and biases dashboard [link](https://wandb.ai/nipunsadvilkar/huggingface/runs/1242bike?workspace=user-nipunsadvilkar
)


#### 2. [iNLTK Marathi news headline classification](https://www.kaggle.com/disisbig/marathi-news-dataset)

This dataset consists 3 classes - `['state', 'entertainment', 'sports']` -  with following docs distribution as per classes:

| train | eval | test
| -- | -- | --
| 9658 | 1210 | 1210

Here as well `roberta-base-mr` outperformed `iNLTK` marathi news text classifier.

Dataset | iNLTK ULMFiT | roberta-base-mr
-- | -- | --
iNLTK news dataset (kaggle) | 92.4 | **94.21**

**🤗 Huggingface Model hub repo:**<br>
`roberta-base-mr` fine tuned on iNLTK news classification dataset model:

[**`flax-community/mr-inltk-classifier`**](https://huggingface.co/flax-community/mr-inltk-classifier)

Fine tuning experiment's weight and biases dashboard [link](https://wandb.ai/nipunsadvilkar/huggingface/runs/2u5l9hon?workspace=user-nipunsadvilkar
)

## **Want to check how above models generalise on real world Marathi data?**

Head to 🤗 Huggingface's spaces 🪐 to play with all three models:
1. Mask Language Modelling with Pretrained Marathi RoBERTa model: <br>
[`flax-community/roberta-base-mr`](https://huggingface.co/flax-community/roberta-base-mr)
2. Marathi Headline classifier: <br>
[**`flax-community/mr-indicnlp-classifier`**](https://huggingface.co/flax-community/mr-indicnlp-classifier)
3. Marathi news classifier: <br>
[**`flax-community/mr-inltk-classifier`**](https://huggingface.co/flax-community/mr-inltk-classifier)

![alt text](https://huggingface.co/docs/assets/hub/icon-space.svg)
[Streamlit app of Pretrained Roberta Marathi model on Huggingface Spaces](https://huggingface.co/spaces/flax-community/roberta-base-mr)



<!-- []https://huggingface.co/spaces/flax-community/roberta-base-mr -->
