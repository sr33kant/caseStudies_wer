import pandas as pd
import ast
import os
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer

df = pd.read_csv("b2b_ner_dataset.csv")
df["tokens_labels"] = df["tokens_labels"].apply(ast.literal_eval)
df["tokens"] = df["tokens_labels"].apply(lambda x: [tok for tok, _ in x])
df["ner_tags"] = df["tokens_labels"].apply(lambda x: [tag for _, tag in x])
df = df.drop(columns=["tokens_labels"])


all_labels = sorted(set(label for tags in df["ner_tags"] for label in tags))
label2id = {label: idx for idx, label in enumerate(all_labels)}
id2label = {idx: label for label, idx in label2id.items()}

df["ner_tags"] = df["ner_tags"].apply(lambda tags: [label2id[t] for t in tags])


dataset = Dataset.from_pandas(df[["tokens", "ner_tags"]])
dataset = dataset.train_test_split(test_size=0.2)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        is_split_into_words=True,
        return_tensors=None
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(i)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./ner-brand-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=1,
    save_steps=500,
    logging_steps=50,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)


trainer.train()
print("Training complete.")
