# Decoding Mental Health States from Social Media Text

Multi-class mental health classifier (Reddit posts → depression / anxiety / PTSD / social anxiety / control) with SHAP explainability scored against clinical instruments (PHQ-9, GAD-7, PCL-5, SPIN).

See `CLAUDE.md` for the full project specification and `paper_draft.md` for the paper draft.

---

## Quick start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd SMA-NLP-Project
pip install -r requirements.txt
```

### 2. Download the dataset

Download the **7 `pre` CSV files** from Zenodo record [3941387](https://zenodo.org/record/3941387) and place them in the project root (same folder as this README):

| File | Label |
|---|---|
| `depression_pre_features_tfidf_256.csv` | depression |
| `anxiety_pre_features_tfidf_256.csv` | anxiety |
| `ptsd_pre_features_tfidf_256.csv` | ptsd |
| `socialanxiety_pre_features_tfidf_256.csv` | social_anxiety |
| `jokes_pre_features_tfidf_256.csv` | control |
| `personalfinance_pre_features_tfidf_256.csv` | control |
| `teaching_pre_features_tfidf_256.csv` | control |

### 3. Run the pipeline

```bash
python run_all.py
```

This runs all steps in order. Total runtime: **~3–5 hours on CPU** (dominated by XGBoost SHAP and DistilBERT training). Steps and estimated times:

| Script | What it does | Time |
|---|---|---|
| `01_pipeline.py` | Merge → deduplicate → downsample → split → scrub | ~2 min |
| `02_train.py` | Train XGBoost on both text arms | ~15 min |
| `02b_evaluate.py` | Save classification results | ~1 min |
| `03_shap.py` | SHAP values for XGBoost | ~15 min |
| `04_lexicons.py` | Build clinical lexicons (PHQ-9/GAD-7/PCL-5/SPIN) | ~1 min |
| `05_alignment.py` | SAS/SPUR/AMB alignment scoring + bootstrap CIs | ~2 min |
| `06_figures.py` | Figures 1–5 (XGBoost results) | ~1 min |
| `08a_bilstm.py` | Fine-tune BiLSTM on both text arms | ~30–60 min CPU |
| `08b_roberta.py` | Fine-tune RoBERTa on both text arms | ~60–90 min CPU |
| `08c_mentalbert.py` | Fine-tune MentalBERT on both text arms | ~60–90 min CPU |
| `09a_roberta_shap.py` | SHAP + alignment for RoBERTa | ~15–30 min CPU |
| `09b_mentalbert_shap.py` | SHAP + alignment for MentalBERT | ~15–30 min CPU |
| `10_figures_multimodel.py` | Figure 6 (model comparison) | ~1 min |

> **GPU strongly recommended for the transformer stages.** Set `CUDA_VISIBLE_DEVICES=0` if available; the scripts auto-detect GPU/MPS/CPU.

---

## What's already in this repo

The following checkpoint files are committed and do **not** need to be regenerated (steps 01–07 already ran):

| File | Contents |
|---|---|
| `model_unscrubbed.pkl` | Trained XGBoost + TF-IDF (unscrubbed arm) |
| `model_scrubbed.pkl` | Trained XGBoost + TF-IDF (scrubbed arm) |
| `shap_data.pkl` | Top-200 SHAP features per class per arm (XGBoost) |
| `lexicons.pkl` | Clinical lexicons: PHQ-9, GAD-7, PCL-5, SPIN |
| `lexicons_for_review.csv` | 165 terms for inter-rater review (see below) |
| `alignment_results.pkl` | SAS/SPUR/AMB scores + bootstrap CIs (XGBoost) |
| `classification_results.pkl` | F1 tables + confusion matrices (XGBoost) |
| `fig1_stacked.png` – `fig5_confusion.png` | Paper figures 1–5 |

**Only steps 08–10 need to be run** (DistilBERT training and comparison figures).

To run only those:

```bash
python 08a_bilstm.py
python 08b_roberta.py
python 08c_mentalbert.py
python 09a_roberta_shap.py
python 09b_mentalbert_shap.py
python 10_figures_multimodel.py
```

---

## Inter-rater lexicon review (human task)

`lexicons_for_review.csv` contains 165 expanded clinical lexicon terms. A second rater needs to add a `rater2` column (1 = valid term, 0 = reject).

Once filled in:

```bash
python run_all.py --all
```

This runs `07_interrater.py` → `05_alignment.py` → `06_figures.py` to compute Cohen's κ, prune the lexicons, and regenerate final scores.

---

## Output files

After the full pipeline, the project root will contain:

```
model_unscrubbed.pkl        XGBoost model (unscrubbed)
model_scrubbed.pkl          XGBoost model (scrubbed)
shap_data.pkl               XGBoost SHAP values
alignment_results.pkl       SAS/SPUR/AMB scores (XGBoost)
classification_results.pkl  F1 + confusion matrices (XGBoost)
roberta_unscrubbed/        RoBERTa model (unscrubbed)
roberta_scrubbed/          RoBERTa model (scrubbed)
mentalbert_unscrubbed/     MentalBERT model (unscrubbed)
mentalbert_scrubbed/       MentalBERT model (scrubbed)
roberta_results.pkl        F1 + confusion matrices (RoBERTa)
mentalbert_results.pkl     F1 + confusion matrices (MentalBERT)
roberta_alignment.pkl      SAS/SPUR/AMB scores (RoBERTa)
mentalbert_alignment.pkl   SAS/SPUR/AMB scores (MentalBERT)
fig1_stacked.png            Alignment decomposition (XGBoost)
fig2_deltas.png             Scrubbing delta: F1 vs SAS
fig3_heatmap.png            Cross-scale alignment matrix
fig4_shap.png               Annotated SHAP bars
fig5_confusion.png          Confusion matrices
fig6_model_comparison.png   XGBoost vs transformer comparison
paper_draft.md              Paper draft
```

---

## Verified numbers (sanity checks)

After `01_pipeline.py`, confirm:
- `merged_raw.csv`: **102,984 rows**
- After dedup: **96,003 rows**
- `merged_balanced.csv`: **70,583 rows**
- `train_df.csv`: **56,465 rows**
- `test_df.csv`: **14,118 rows**

Any difference → check pandas version or CSV file versions before continuing.

---

## Environment

- Python 3.10+
- Originally developed in Google Colab; fully runnable locally
- All random seeds: `random_state=42`
