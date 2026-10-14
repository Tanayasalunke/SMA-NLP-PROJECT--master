import os, pickle, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import ssl
import urllib3
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()
import requests
from requests.sessions import Session
original_request = Session.request
def new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)
Session.request = new_request

BASE = os.path.dirname(os.path.abspath(__file__))

MAX_LEN   = 128
EPOCHS    = 3
BATCH     = 16
LR        = 2e-5
MAX_TRAIN = 40000   # cap rows for tractable training; None = use all 56,465

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        EvalPrediction,
    )
except ImportError:
    raise SystemExit(
        "Missing dependencies. Run:\n"
        "  pip install transformers torch accelerate"
    )


class MentalHealthDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"macro_f1": f1_score(p.label_ids, preds, average="macro")}


def train_arm(arm, text_col, train_df, test_df, le):
    print(f"\n{'='*60}")
    print(f"ARM: {arm}  (column: {text_col})")
    print("="*60)

    num_labels = len(le.classes_)
    id2label   = {i: c for i, c in enumerate(le.classes_)}
    label2id   = {c: i for i, c in enumerate(le.classes_)}

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Proportionally subsample training rows if capped
    tr = train_df
    if MAX_TRAIN and len(tr) > MAX_TRAIN:
        tr = (
            tr.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(
                n=min(len(x), MAX_TRAIN // num_labels),
                random_state=42
            ))
            .reset_index(drop=True)
        )
        print(f"  Subsampled train: {len(tr)} rows (from {len(train_df)})")

    y_train = le.transform(tr["label"])
    y_test  = le.transform(test_df["label"])

    print(f"  Tokenizing {len(tr)} train + {len(test_df)} test rows...")
    t0 = time.time()
    train_enc = tokenizer(
        tr[text_col].fillna("").tolist(),
        truncation=True, padding="max_length", max_length=MAX_LEN
    )
    test_enc = tokenizer(
        test_df[text_col].fillna("").tolist(),
        truncation=True, padding="max_length", max_length=MAX_LEN
    )
    print(f"  Tokenized in {time.time()-t0:.0f}s")

    train_ds = MentalHealthDataset(train_enc, y_train)
    test_ds  = MentalHealthDataset(test_enc,  y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    out_dir = os.path.join(BASE, f"roberta_{arm}")
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=LR,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        fp16=False,   # mixed precision causes issues on MPS for some ops
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print(f"  Training {EPOCHS} epochs on "
          f"{'GPU' if torch.cuda.is_available() else 'CPU'}...")
    t1 = time.time()
    trainer.train()
    elapsed = time.time() - t1
    print(f"  Training done in {elapsed/60:.1f} min")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"  Model saved to {out_dir}/")

    preds_out = trainer.predict(test_ds)
    y_pred    = np.argmax(preds_out.predictions, axis=1)

    report = classification_report(
        y_test, y_pred, target_names=le.classes_,
        output_dict=True, digits=3
    )
    cm = confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
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

    out = os.path.join(BASE, "classification_results_roberta.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {out}")
    print("Run SHAP extraction next.")


if __name__ == "__main__":
    main()
