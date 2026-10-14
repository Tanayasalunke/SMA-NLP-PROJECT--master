"""
Run SHAP on both model arms.
Saves:
  - mean |SHAP| per feature per class (all features) -- for main alignment scoring
  - per-sample |SHAP| for top-200 features per class  -- for bootstrapping
Output: shap_data.pkl
Runtime: ~5-10 min per arm (2000-row sample).
"""
import os
import pickle
import numpy as np
import pandas as pd
import shap

BASE    = os.path.dirname(os.path.abspath(__file__))
N_SHAP  = 2000
K_SAVE  = 200    # save top-200 per class for bootstrap sensitivity up to k=100
SEED    = 42


def load_model(fname):
    with open(os.path.join(BASE, fname), "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["vectorizer"], saved["label_encoder"]


def run_shap(model, vectorizer, le, test_df, text_col):
    sample       = test_df.sample(n=N_SHAP, random_state=SEED)
    X_sample     = vectorizer.transform(sample[text_col])
    feature_names = vectorizer.get_feature_names_out()
    classes       = list(le.classes_)

    print(f"  Running TreeExplainer on {N_SHAP} samples x {X_sample.shape[1]} features...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # shap_values shape: (n_samples, n_features, n_classes)

    arm_data = {"classes": classes}
    for i, cls in enumerate(classes):
        sv_cls      = np.abs(shap_values[:, :, i])          # (2000, 10000)
        mean_abs    = sv_cls.mean(axis=0)                    # (10000,)

        top200_idx  = np.argsort(mean_abs)[::-1][:K_SAVE]
        top_names   = feature_names[top200_idx]
        top_mean    = mean_abs[top200_idx]
        top_samples = sv_cls[:, top200_idx].astype(np.float32)  # (2000, 200)

        arm_data[cls] = {
            "top_feature_names": top_names,      # (200,) array
            "top_mean_abs_shap": top_mean,        # (200,) array
            "top_sample_shap":   top_samples,     # (2000, 200) array — for bootstrap
        }
        print(f"  [{cls}] top feature: '{top_names[0]}' (mean |SHAP|={top_mean[0]:.4f})")

    return arm_data


def main():
    test_df = pd.read_csv(os.path.join(BASE, "test_df.csv"))
    results = {}

    for arm, fname, text_col in [
        ("unscrubbed", "model_unscrubbed.pkl", "text_unscrubbed"),
        ("scrubbed",   "model_scrubbed.pkl",   "text_scrubbed"),
    ]:
        print(f"\n{'='*60}\nSHAP -- {arm} arm\n{'='*60}")
        model, vec, le = load_model(fname)
        results[arm]   = run_shap(model, vec, le, test_df, text_col)

    out = os.path.join(BASE, "shap_data.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {out}")
    print("SHAP complete. Run 05_alignment.py next.")


if __name__ == "__main__":
    main()
