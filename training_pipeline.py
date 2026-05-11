import torch
import pandas as pd
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

LABEL_MAP = {"bad": 0, "medium": 1, "good": 2}
LABEL_NAMES = ["bad", "medium", "good"]


def data_split(data_path: Path, seed: int):
    '''Generates the X and Y split of the dataset'''
    df = pd.read_csv(data_path)

    X = df["review"]
    y = df["label"].map(LABEL_MAP)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test


class DrugReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len = 256):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length = self.max_len,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class DrugReviewClassifier(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.1):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size   # 756 or 1024

        # simple classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        '''mean pool all token embeddings'''
        mask_expanded = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        '''pass text embeddings through classifier network'''
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        return logits


def train_epoch(model, loader, optimizer, scheduler, criterion, device = "cuda"):
    model.train() # set model in training mode
    total_loss, all_preds, all_labels = 0, [], []

    for batch in loader:
        # get ids, attn masks and labels for the batch
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # reset gradients
        optimizer.zero_grad()
        
        # obtain logits and the loss
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # update gradient values
        loss.backward()

        # limit grad values (so steps are not too large)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        
        # step the optim and sched
        optimizer.step()
        scheduler.step()

        # update total loss and predictions
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # obtain average loss and f1 for the epoch
    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


def eval_epoch(model, loader, criterion, device):
    model.eval()    # set model in eval mode
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in loader:
            # obtain ids attn mask and labels
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # get logits and their loss
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # 
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_labels


def train_model(
    model_name: str,
    data_path: Path,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 128,
    dropout: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # split train test ds
    X_train, X_test, y_train, y_test = data_split(data_path, seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = DrugReviewDataset(X_train, y_train, tokenizer, max_len)
    test_dataset = DrugReviewDataset(X_test, y_test, tokenizer, max_len)

    # create loaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    model = DrugReviewClassifier(model_name, num_labels = 3, dropout = dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr * 10},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = warmup_steps,
        num_training_steps = total_steps,
    )

    best_f1, best_state = 0, None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_f1, preds, gt = eval_epoch(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f}  F1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f}  F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best model (F1={best_f1:.4f})")

    # save best state
    model.load_state_dict(best_state)
    
    # validate on test set
    _, _, preds, gt = eval_epoch(model, test_loader, criterion, device)
    print("\nFinal Classification Report:")
    print(classification_report(gt, preds, target_names=LABEL_NAMES))

    # plot loss 
    plt.figure()
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train loss")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, tokenizer


if __name__ == "__main__":
    train_model()