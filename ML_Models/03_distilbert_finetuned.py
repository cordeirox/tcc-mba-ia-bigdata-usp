import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import classification_report

df = pd.read_csv("data/Letras-e-generos-full.csv", encoding='latin-1', sep=';', engine='python')
df.columns = df.columns.str.strip()
df['Music Genre'] = df['Music Genre'].str.lower().str.strip()
df = df.drop_duplicates(subset='Lyrics')
df_balanced = df.groupby('Music Genre', group_keys=False).apply(lambda x: x.sample(n=300, random_state=42)).reset_index(drop=True)

X = df_balanced['Lyrics']
y = LabelEncoder().fit_transform(df_balanced['Music Genre'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_enc = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
test_enc = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

class DatasetHF(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {'labels': torch.tensor(self.labels[idx])}
    def __len__(self):
        return len(self.labels)

train_ds = DatasetHF(train_enc, y_train)
test_ds = DatasetHF(test_enc, y_test)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(y)))

training_args = TrainingArguments(output_dir='./results', per_device_train_batch_size=8, num_train_epochs=3,
                                  evaluation_strategy="epoch", logging_steps=10, save_strategy="no", report_to="none")

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds)
trainer.train()

preds = trainer.predict(test_ds).predictions
y_pred = np.argmax(preds, axis=1)

print(classification_report(y_test, y_pred))
