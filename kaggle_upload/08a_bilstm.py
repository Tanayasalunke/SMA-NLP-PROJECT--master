import os
import pickle
import time
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters
MAX_LEN = 128
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.5
MAX_TRAIN = 40000

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

def tokenizer(text):
    text = str(text).lower()
    return re.findall(r'\b\w+\b', text)

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        self.unk_idx = word2idx.get("<unk>", 1)
        self.pad_idx = word2idx.get("<pad>", 0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        tokens = tokenizer(text)
        indices = [self.word2idx.get(w, self.unk_idx) for w in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [self.pad_idx] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

def train_arm(arm, text_col, train_df, test_df, le):
    print(f"\n{'='*60}")
    print(f"ARM: {arm}  (column: {text_col})")
    print("="*60)

    num_classes = len(le.classes_)

    tr = train_df
    if MAX_TRAIN and len(tr) > MAX_TRAIN:
        tr = (
            tr.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(
                n=min(len(x), MAX_TRAIN // num_classes),
                random_state=42
            ))
            .reset_index(drop=True)
        )
        print(f"  Subsampled train: {len(tr)} rows (from {len(train_df)})")

    y_train = le.transform(tr["label"])
    y_test  = le.transform(test_df["label"])

    print("  Loading GloVe embeddings using gensim...")
    import gensim.downloader as api
    print("  Loading GloVe (glove-wiki-gigaword-300)...")
    glove = api.load("glove-wiki-gigaword-300")
    
    print("  Building vocabulary...")
    word2idx = {"<pad>": 0, "<unk>": 1}
    idx2word = {0: "<pad>", 1: "<unk>"}
    
    pretrained_embeddings = [np.zeros(EMBEDDING_DIM), np.random.normal(scale=0.1, size=(EMBEDDING_DIM,))]
    
    idx = 2
    for w in glove.index_to_key:
        word2idx[w] = idx
        idx2word[idx] = w
        pretrained_embeddings.append(glove[w])
        idx += 1
        
    pretrained_embeddings = np.array(pretrained_embeddings)
    pretrained_embeddings = torch.tensor(pretrained_embeddings, dtype=torch.float32)
    
    print(f"  Vocab size: {len(word2idx)}.")

    train_ds = TextDataset(tr[text_col], y_train, word2idx, MAX_LEN)
    test_ds = TextDataset(test_df[text_col], y_test, word2idx, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTMClassifier(
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        padding_idx=word2idx["<pad>"]
    )
    
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.to(device)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"  Training for {EPOCHS} epochs on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        t0 = time.time()
        for batch_texts, batch_labels in train_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            predictions = model(batch_texts)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"    Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-t0:.1f}s")

    print("  Evaluating...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_texts, _ in test_loader:
            batch_texts = batch_texts.to(device)
            predictions = model(batch_texts)
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    y_pred = np.array(all_preds)
    
    report = classification_report(
        y_test, y_pred, target_names=le.classes_,
        output_dict=True, digits=3
    )
    cm = confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
    
    out_dir = os.path.join(BASE, f"bilstm_{arm}.pt")
    torch.save(model.state_dict(), out_dir)
    print(f"  Model saved to {out_dir}")
    
    return {"report": report, "cm": cm, "classes": list(le.classes_)}


def main():
    print("Loading train/test data...")
    train_df = pd.read_csv(os.path.join(BASE, "train_df.csv"))
    test_df  = pd.read_csv(os.path.join(BASE, "test_df.csv"))

    le = LabelEncoder()
    le.fit(train_df["label"])
    print(f"Classes: {list(le.classes_)}")

    results = {}
    for arm, col in [("unscrubbed", "text_unscrubbed"),
                     ("scrubbed",   "text_scrubbed")]:
        results[arm] = train_arm(arm, col, train_df, test_df, le)

    out = os.path.join(BASE, "classification_results_bilstm.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
