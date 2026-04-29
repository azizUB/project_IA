import torch
import pandas as pd
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel
from sklearn.model_selection import train_test_split


def data_split(data_path: Path, seed: int):
    "split dataset into train and "
    
    # load data df
    df = pd.read_csv(data_path)
    
    # split features and target
    X = df['review']
    y = df['label']
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = seed, stratify = y
    )
    
    return X_train, X_test, y_train, y_test

# classification model
class DrugReviewClassifier(nn.Module):
    def __init__(self, model_name, num_labels = 3, dropout = 0.1):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size   # embedding size (768 or 1024)
        
        # create classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        
    def mean_pooling(self, token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        return logits