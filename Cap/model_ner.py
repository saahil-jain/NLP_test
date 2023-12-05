import ast
from tqdm import tqdm
tqdm.pandas()

import pandas as pd
import numpy as np

import torch

import evaluate
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)

import nltk
nltk.download('punkt')
from datasets import load_dataset
from datasets import Dataset

def to_list(sample):
    sample['tokens'] = ast.literal_eval(sample['tokens'])
    sample['labels'] = ast.literal_eval(sample['labels'])
    return sample

seed = 1
data = load_dataset('csv', data_files='data.csv')
data = data.map(to_list)
data = data['train'].train_test_split(test_size=0.2, seed=seed)
data

label_names = ['O', 'B-gene', 'B-both', 'B-onco', 'B-tsg']

idx = 0

words = data['train'][idx]['tokens']
labels = data['train'][idx]['labels']
line1 = ''
line2 = ''
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)

model_checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.is_fast

inputs = tokenizer(data['train'][0]['tokens'], is_split_into_words=True)
print(inputs.tokens())

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

labels = data['train'][0]['labels']
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, is_split_into_words=True
    )
    all_labels = examples['labels']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs

tokenized_datasets = data.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=data['train'].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

metric = evaluate.load('seqeval')

labels = data['train'][0]['labels']
labels = [label_names[i] for i in labels]
print(labels)

predictions = labels.copy()
predictions[2] = 'B-onco'
metric.compute(predictions=[predictions], references=[labels])

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

model.config.num_labels

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()