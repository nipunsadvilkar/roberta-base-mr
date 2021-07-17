"""
iNLTK public marathi headlines dataset
"""
import pandas as pd
import numpy as np
from datasets import load_metric

from datasets import Dataset
from datasets import ClassLabel
from transformers import TrainingArguments, Trainer, AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MAX_LEN = 128
MODEL_NAME = "flax-community/roberta-base-mr"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["headline"], padding="max_length", truncation=True, max_length=MAX_LEN)


train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")

label_names = train_df["label"].unique().tolist()
num_labels = len(label_names)
cl = ClassLabel(num_classes=num_labels, names=label_names)
valid_df["label"] = valid_df["label"].map(lambda x: cl.str2int(x))
train_df["label"] = train_df["label"].map(lambda x: cl.str2int(x))

print(label_names)

label2id = {label : cl.str2int(label) for label in label_names}
id2label = {cl.str2int(label) : label for label in label_names}

print(label2id)

config = AutoConfig.from_pretrained(MODEL_NAME, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, from_flax=True, config=config)

train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)

valid_tokenized_data = valid_ds.map(tokenize_function, batched=True)
train_tokenized_data = train_ds.map(tokenize_function, batched=True)

training_args = TrainingArguments("inltk_trainer", report_to=None)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_data,
    eval_dataset=valid_tokenized_data,
    compute_metrics=compute_metrics,
)
trainer.train()

model.save_pretrained("inltk-mr-classifier")
tokenizer.save_pretrained("inltk-mr-classifier")

trainer.evaluate()