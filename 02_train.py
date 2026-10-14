"""
Train TF-IDF + XGBoost on both text arms (unscrubbed and scrubbed).
Outputs: model_unscrubbed.pkl, model_scrubbed.pkl
Runtime: ~6-10 min per arm on CPU.
"""
import os
import pickle
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))


def load_data():
    train_df = pd.read_csv(os.path.join(BASE, "train_df.csv"))
    test_df  = pd.read_csv(os.path.join(BASE, "test_df.csv"))
    return train_df, test_df


def make_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 1),
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
    )


def make_model(num_class):
    return XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.15,
        tree_method="hist",
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def train_arm(arm_name, text_col, train_df, test_df, y_train, y_test, le, sample_weights):
    print(f"\n{'='*60}")
    print(f"ARM: {arm_name}  (column: {text_col})")
    print('='*60)

    t0 = time.time()
    vec = make_vectorizer()
    X_train = vec.fit_transform(train_df[text_col])
    X_test  = vec.transform(test_df[text_col])
    print(f"Vectorized in {time.time()-t0:.1f}s | shape: {X_train.shape}")

    t1 = time.time()
    model = make_model(num_class=len(le.classes_))
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"Trained in {time.time()-t1:.1f}s")

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=le.classes_, digits=3))

    return model, vec


def main():
    print("Loading train/test data...")
    train_df, test_df = load_data()

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test  = le.transform(test_df["label"])
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    print(f"Classes: {list(le.classes_)}")

    model_u, vec_u = train_arm(
        "unscrubbed", "text_unscrubbed",
        train_df, test_df, y_train, y_test, le, sample_weights
    )
    pkl_u = os.path.join(BASE, "model_unscrubbed.pkl")
    with open(pkl_u, "wb") as f:
        pickle.dump({"model": model_u, "vectorizer": vec_u, "label_encoder": le}, f)
    print(f"Saved {pkl_u}")

    model_s, vec_s = train_arm(
        "scrubbed", "text_scrubbed",
        train_df, test_df, y_train, y_test, le, sample_weights
    )
    pkl_s = os.path.join(BASE, "model_scrubbed.pkl")
    with open(pkl_s, "wb") as f:
        pickle.dump({"model": model_s, "vectorizer": vec_s, "label_encoder": le}, f)
    print(f"Saved {pkl_s}")

    # Save classification results so downstream scripts don't hardcode F1 values
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "evaluate", os.path.join(BASE, "02b_evaluate.py")
    )
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    ev.main()

    print("\nTraining complete. Run 03_shap.py next.")


if __name__ == "__main__":
    main()
