# utils/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

def apk(actual, predicted, k=3):
    """Calculates Average Precision at k."""
    predicted = predicted[:k]
    for i, p in enumerate(predicted):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0

def mapk(actuals, predictions, k=3):
    """Calculates Mean Average Precision at k."""
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)])

def make_top3_predictions(df, score_col):
    """Groups options by sample ID and extracts the top 3 ranked option labels."""
    ranked = (
        df.sort_values(["id", score_col], ascending=[True, False])
          .groupby("id")["option_label"]
          .apply(list)
          .reset_index()
    )
    ranked["top3"] = ranked["option_label"].apply(lambda x: x[:3])
    return ranked[["id", "top3"]]

def stable_softmax(x):
    """Computes softmax values safely by preventing numerical overflow."""
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    """Computes classification metrics for binary/long-format targets using Macro F1."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "loss": log_loss(y_true, y_prob, labels=[0, 1])
    }

def compute_multiclass_metrics(y_true, logits):
    """Computes metrics from multiclass raw model outputs using Macro F1."""
    probs = stable_softmax(logits)
    y_pred = np.argmax(probs, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "loss": log_loss(y_true, probs, labels=[0, 1, 2, 3, 4])
    }
    return metrics, y_pred, probs

# utils/datasets.py
import re
import torch
import pandas as pd
from torch.utils.data import Dataset

def clean_text(text):
    """Applies standard lowering, regex HTML stripping, and character filtering."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)                   
    text = re.sub(r"http\S+|www\S+", " ", text)           
    text = re.sub(r"[^a-z0-9\s\-\?\!\.,:;]", " ", text)    
    text = re.sub(r"\s+", " ", text).strip()              
    return text

def build_long_format(df, is_train=True):
    """Transforms standard multiple choice rows into structured long-format rows."""
    rows = []
    labels = ["A", "B", "C", "D", "E"]
    for _, row in df.iterrows():
        for label in labels:
            combined_text = f"question: {row['prompt']} [SEP] option: {row[label]}"
            item = {
                "id": row.get("id", None),
                "prompt": row["prompt"],
                "option_label": label,
                "option_text": row[label],
                "retrieved_context": "", 
                "combined_text": combined_text
            }
            if is_train and "answer" in row:
                item["target"] = 1 if row["answer"] == label else 0
            rows.append(item)
    return pd.DataFrame(rows)

class ScratchDataset(Dataset):
    """Custom PyTorch Dataset helper for scratch models (BiGRU)."""
    def __init__(self, df, vocab, max_len=128, is_train=True):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.is_train = is_train

    def _simple_tokenize(self, text):
        return str(text).split()

    def _encode_text(self, text):
        tokens = self._simple_tokenize(text)
        ids = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in tokens[:self.max_len]]
        if len(ids) < self.max_len:
            ids = ids + [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {
            'input_ids': torch.tensor(self._encode_text(row['combined_text']), dtype=torch.long),
            'id': row['id'],
            'option_label': row['option_label']
        }
        if self.is_train:
            item['target'] = torch.tensor(row['target'], dtype=torch.float)
        return item

class ElectraMCQDataset(Dataset):
    """Custom PyTorch Dataset helper for HuggingFace multiple choice transformer inputs."""
    def __init__(self, df, tokenizer, max_len=320, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.label2id = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row["prompt"]).strip()
        question_text = f"Question:\n{prompt}\n\nTask:\nSelect the most appropriate answer."
        options = [str(row[label]).strip() for label in ["A", "B", "C", "D", "E"]]

        first_sentences = [question_text.strip()] * 5
        second_sentences = [f"Candidate Answer:\n{option}".strip() for option in options]

        encoded = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        if self.is_train:
            ans_key = row["answer"]
            ans_id = self.label2id.get(ans_key, 0)
            item["labels"] = torch.tensor(ans_id, dtype=torch.long)

        if "id" in row and pd.notna(row["id"]):
            try:
                item["id"] = torch.tensor(int(row["id"]), dtype=torch.long)
            except (ValueError, TypeError):
                item["id"] = torch.tensor(idx, dtype=torch.long)
        return item

    