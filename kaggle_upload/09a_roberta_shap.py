import os, pickle, re, time
import numpy as np
import pandas as pd

BASE        = os.path.dirname(os.path.abspath(__file__))
SHAP_SAMPLE_PER_CLASS = 100    # Set to 100 to keep runtime manageable while still stable
K_PRIMARY   = 50

try:
    import torch, shap
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
except ImportError:
    raise SystemExit("Run: pip install transformers torch shap")

import nltk; nltk.download("wordnet", quiet=True)
from nltk.stem import WordNetLemmatizer
LEM = WordNetLemmatizer()


def lemmatize(w):
    w = LEM.lemmatize(w.lower(), "v")
    return LEM.lemmatize(w, "n")


def load(fname):
    with open(os.path.join(BASE, fname), "rb") as f:
        return pickle.load(f)


def make_predictor(arm, model_name="roberta"):
    model_dir = os.path.join(BASE, f"{model_name}_{arm}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    device = torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    classes   = list(model.config.id2label[i] for i in range(model.config.num_labels))

    def predict_proba(texts):
        enc = tokenizer(
            list(texts),
            truncation=True, padding=True,
            max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    return predict_proba, classes, tokenizer


def aggregate_to_words(shap_values_obj):
    n_cls = shap_values_obj[0].values.shape[-1]
    word_shap = [{} for _ in range(n_cls)]

    for i in range(len(shap_values_obj)):
        tokens = shap_values_obj[i].data      # list of string tokens
        vals   = shap_values_obj[i].values    # (n_tokens, n_classes)

        for tok, val_row in zip(tokens, vals):
            # Strip RoBERTa/BPE specific prefixes (e.g. 'Ġ')
            clean = re.sub(r"^[^\w]+", "", tok.lower())
            clean = re.sub(r"[^\w]", "", clean)
            if not clean or clean in {"cls", "sep", "pad", "unk", "s"}:
                continue
            for ci in range(n_cls):
                word_shap[ci][clean] = word_shap[ci].get(clean, 0.0) + abs(val_row[ci])

    n = len(shap_values_obj)
    for ci in range(n_cls):
        for w in word_shap[ci]:
            word_shap[ci][w] /= n

    return word_shap


def compute_shap_arm(arm, test_df, text_col, model_name="roberta"):
    print(f"\n  ARM: {arm}")
    predict_proba, classes, tokenizer = make_predictor(arm, model_name)

    masker    = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker, output_names=classes)

    # Sample SHAP_SAMPLE_PER_CLASS from each class
    sampled_df = test_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), SHAP_SAMPLE_PER_CLASS), random_state=42)
    ).reset_index(drop=True)
    
    sample = sampled_df[text_col].fillna("").tolist()

    print(f"  Running SHAP on {len(sample)} samples...")
    t0 = time.time()
    # reduce max_evals to speed up
    shap_vals = explainer(sample, max_evals=100, batch_size=32)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

    word_shap = aggregate_to_words(shap_vals)

    arm_data = {}
    for ci, cls in enumerate(classes):
        sorted_feats = sorted(word_shap[ci].items(),
                              key=lambda x: x[1], reverse=True)[:200]
        arm_data[cls] = {
            "top_feature_names": np.array([f for f, _ in sorted_feats]),
            "top_mean_abs_shap": np.array([v for _, v in sorted_feats],
                                           dtype=np.float32),
        }
        top = sorted_feats[0] if sorted_feats else ("—", 0.0)
        print(f"    {cls}: top = '{top[0]}' ({top[1]:.4f})")

    return arm_data


CLASS_TO_INSTR = {
    "depression": "PHQ9", "anxiety": "GAD7",
    "ptsd": "PCL5", "social_anxiety": "SPIN", "control": None,
}

def make_set(terms):
    s = set()
    for t in terms:
        s.add(t.lower())
        s.add(lemmatize(t))
    return s


def score_alignment(arm_data, lexicons, artifact_set, k=K_PRIMARY):
    lex_sets = {k_: make_set(v) for k_, v in lexicons.items()}
    art_set  = make_set(list(artifact_set))

    scores = {}
    for cls, data in arm_data.items():
        names = data["top_feature_names"][:k]
        phi   = data["top_mean_abs_shap"][:k]
        total = phi.sum() if len(phi) > 0 else 0

        if total == 0 or len(names) == 0:
            scores[cls] = {"SAS": 0.0, "SPUR": 0.0, "AMB": 1.0}
            continue

        instr = CLASS_TO_INSTR.get(cls)
        L = lex_sets.get(instr, set()) if instr else set()
        A = art_set

        m_L = np.array([1.0 if (n.lower() in L or lemmatize(n) in L) else 0.0
                         for n in names])
        m_A = np.array([1.0 if (n.lower() in A or lemmatize(n) in A) else 0.0
                         for n in names])

        sas  = float((phi * m_L).sum() / total)
        spur = float((phi * m_A * (1 - m_L)).sum() / total)
        amb  = 1.0 - sas - spur
        scores[cls] = {"SAS": sas, "SPUR": spur, "AMB": amb}

    return scores


def main():
    test_df      = pd.read_csv(os.path.join(BASE, "test_df.csv"))
    lex_d        = load("lexicons.pkl")
    lexicons     = lex_d["lexicons"]
    artifact_set = set(lex_d["artifact_list"])

    shap_out  = {}
    align_out = {}

    for arm, col in [("unscrubbed", "text_unscrubbed"),
                     ("scrubbed",   "text_scrubbed")]:
        arm_data = compute_shap_arm(arm, test_df, col, model_name="roberta")
        shap_out[arm] = arm_data

        print(f"\n  Alignment scores ({arm}):")
        align_out[arm] = score_alignment(arm_data, lexicons, artifact_set)
        for cls, s in align_out[arm].items():
            print(f"    {cls}: SAS={s['SAS']:.4f} SPUR={s['SPUR']:.4f} "
                  f"AMB={s['AMB']:.4f}")

    with open(os.path.join(BASE, "roberta_shap_data.pkl"), "wb") as f:
        pickle.dump(shap_out, f)
    with open(os.path.join(BASE, "roberta_alignment.pkl"), "wb") as f:
        pickle.dump(align_out, f)

    print("\nSaved roberta_shap_data.pkl and roberta_alignment.pkl")


if __name__ == "__main__":
    main()
