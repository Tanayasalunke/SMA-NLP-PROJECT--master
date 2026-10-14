"""
Load saved models and produce classification_results.pkl.
Run this once after 02_train.py if you need results without retraining.
02_train.py calls this automatically going forward.
"""
import os, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))


def evaluate(arm, model_file, text_col, test_df):
    with open(os.path.join(BASE, model_file), "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    vec   = saved["vectorizer"]
    le    = saved["label_encoder"]

    X = vec.transform(test_df[text_col])
    y_true = le.transform(test_df["label"])
    y_pred = model.predict(X)

    report = classification_report(
        y_true, y_pred, target_names=le.classes_,
        output_dict=True, digits=3
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{arm.upper()}")
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=3))

    return {
        "report":  report,          # dict: class -> {precision, recall, f1-score, support}
        "cm":      cm,              # ndarray (n_classes x n_classes)
        "classes": list(le.classes_),
    }


def main():
    test_df = pd.read_csv(os.path.join(BASE, "test_df.csv"))

    results = {
        "unscrubbed": evaluate("unscrubbed", "model_unscrubbed.pkl", "text_unscrubbed", test_df),
        "scrubbed":   evaluate("scrubbed",   "model_scrubbed.pkl",   "text_scrubbed",   test_df),
    }

    out = os.path.join(BASE, "classification_results.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {out}")
    return results


if __name__ == "__main__":
    main()
