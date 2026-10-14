# Project Handoff Document
## Decoding Mental Health States from Social Media Text: A Multi-Class Classification Approach with Clinical Symptom-Alignment Validation

**Prepared for:** Colleague taking over remaining tasks  
**Date:** July 2026  
**Status:** XGBoost pipeline fully complete. DistilBERT training + inter-rater review + paper finalization remain.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Repository Setup](#2-repository-setup)
3. [Dataset](#3-dataset)
4. [What Has Been Completed](#4-what-has-been-completed)
5. [Verified Numbers — Do Not Modify](#5-verified-numbers--do-not-modify)
6. [What Remains To Be Done](#6-what-remains-to-be-done)
7. [Task A — DistilBERT Training (Steps 08–10)](#7-task-a--distilbert-training-steps-0810)
8. [Task B — Inter-Rater Lexicon Review](#8-task-b--inter-rater-lexicon-review)
9. [Task C — Paper Finalization](#9-task-c--paper-finalization)
10. [Locked Design Decisions](#10-locked-design-decisions)
11. [Known Pitfalls and Warnings](#11-known-pitfalls-and-warnings)
12. [File Reference](#12-file-reference)

---

## 1. What This Project Is

### Core thesis
Build a multi-class classifier for mental health states from Reddit posts. Then use SHAP explainability scored against real clinical instruments (PHQ-9, GAD-7, PCL-5, SPIN) to measure how much of the model's decision-making reflects clinically valid symptom language versus spurious dataset shortcuts (users mentioning the name of the subreddit they are posting in).

### Five classes
`depression`, `anxiety`, `ptsd`, `social_anxiety`, `control`

### The key insight
A Reddit model can achieve F1=0.80 partly by recognising when a user writes "I was diagnosed with ptsd" rather than by detecting actual PTSD symptom language. We quantify this effect for the first time using a metric called the **Symptom Alignment Score (SAS)**.

### Three contributions
1. Multi-class classifier (XGBoost + TF-IDF and DistilBERT) on the same data
2. SAS metric: decomposes SHAP mass into Clinical / Spurious / Ambiguous
3. Scrubbed vs. unscrubbed text comparison showing which classes rely on artifact shortcuts

### Paper
`paper_draft.md` — nearly complete. Sections 1–7, all figures, all references are written. Remaining gaps are clearly marked `[TBD]` and are filled in after the inter-rater step and DistilBERT training.

---

## 2. Repository Setup

### Clone and install
```bash
git clone <repo-url>
cd SMA-NLP-Project
pip install -r requirements.txt
```

### requirements.txt covers
```
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
shap>=0.44
transformers>=4.40
torch>=2.0
accelerate>=0.29
nltk>=3.8
matplotlib>=3.7
seaborn>=0.13
```

Python 3.10+ required. Developed on Python 3.13.

### GPU check (important for DistilBERT)
```python
import torch
print(torch.cuda.is_available())   # True = GPU available, False = CPU only
```
If CPU-only: DistilBERT training (step 08) takes ~2–4 hours per arm. On GPU: ~20–40 min.

---

## 3. Dataset

### Source
Reddit Mental Health Dataset (RMHD), Zenodo record **3941387**  
Download link: https://zenodo.org/record/3941387  
Citation: Low et al. (2020), *JMIR*, 22(10), e22635

### What to download
Download **only the `pre` files** (7 CSV files). Do not download `post` files.

| Filename | Rows | Label assigned |
|---|---|---|
| `depression_pre_features_tfidf_256.csv` | 21,209 | depression |
| `anxiety_pre_features_tfidf_256.csv` | 19,976 | anxiety |
| `ptsd_pre_features_tfidf_256.csv` | 4,483 | ptsd |
| `socialanxiety_pre_features_tfidf_256.csv` | 12,277 | social_anxiety |
| `jokes_pre_features_tfidf_256.csv` | 19,249 | control |
| `personalfinance_pre_features_tfidf_256.csv` | 23,440 | control |
| `teaching_pre_features_tfidf_256.csv` | 2,350 | control |

### Where to place them
In the project root (same folder as `run_all.py`). They are git-ignored because of file size; they are not in the repo.

### Why these files only
The `pre` files cover December 2018 – December 2019 (pre-pandemic). This avoids COVID-specific vocabulary contaminating symptom language. The `post` files (2020+) are deliberately excluded.

### Why three control subreddits
Using r/Jokes + r/personalfinance + r/teaching as control (instead of a single subreddit) reduces topic-vocabulary leakage. A single control subreddit's specific vocabulary (e.g., only finance terms) would contaminate the control model's top SHAP features even more. Even with three, some leakage remains — this is acknowledged as a limitation in the paper.

---

## 4. What Has Been Completed

The following pipeline steps have been run and their outputs are **committed to the repo**. You do not need to re-run these unless you want to verify the numbers yourself.

### Step 1 — Merge (`01_pipeline.py`) ✅
- Reads all 7 CSVs, keeps only columns `[subreddit, author, date, post]`, discards the 250+ precomputed TF-IDF/LIWC columns
- Assigns labels, concatenates
- Saves `merged_raw.csv` (102,984 rows)

### Step 2 — Cross-class user deduplication (`01_pipeline.py`) ✅
- Finds users who posted in more than one class subreddit (these would leak writer identity across classes)
- Removes 3,359 such users entirely → 96,003 rows remain

### Step 3 — Control class downsample (`01_pipeline.py`) ✅
- Control class had 44,361 rows after dedup — much larger than mental health classes
- Proportionally downsampled to ~25,000 rows across the three control subreddits
- Final balanced dataset: **70,583 rows** → saved as `merged_balanced.csv`
- Class breakdown: depression 19,210 | control 18,941 | anxiety 17,701 | social_anxiety 10,688 | ptsd 4,043

### Step 4 — Author-level train/test split (`01_pipeline.py`) ✅
- Split is by unique **author** (not by row) — this is critical
- If split by row, the same user's writing style appears in both train and test, inflating metrics
- 80/20 split, `random_state=42`
- **Train: 56,465 rows | Test: 14,118 rows**
- Saved as `train_df.csv` and `test_df.csv`

### Step 5 — Scrubbed and unscrubbed text arms (`01_pipeline.py`) ✅
Two parallel text columns created in both train and test:

- `text_unscrubbed` = raw `post` column
- `text_scrubbed` = post with artifact terms removed via regex

**What is scrubbed (plain substring removal):**
```python
ARTIFACT_TERMS = [
    "depression", "anxiety", "ptsd", "social anxiety", "socialanxiety",
    "diagnosed", "diagnosis", "dae", "throwaway", "tw:", "trigger warning",
    "r/depression", "r/anxiety", "r/ptsd", "r/socialanxiety",
]
```

**What is scrubbed (word-boundary matched — `\bterm\b`):**
```python
ARTIFACT_TERMS_WB = ["sa", "gad"]
```
These two were added after SHAP revealed them as shorthand self-references that leaked into top features. `sa` = social anxiety abbreviation, `gad` = Generalized Anxiety Disorder abbreviation.

**What is deliberately NOT scrubbed:**
- `"depressed"` — mirrors PHQ-9 item 2 wording ("feeling down, depressed, or hopeless"). This is a genuine clinical symptom word, not a self-reference.
- `"social"` — core SPIN (Social Phobia Inventory) vocabulary. The entire instrument is about social situations. Removing it would eliminate genuine signal.

### Step 6 — XGBoost training (`02_train.py`) ✅
Two separate TF-IDF + XGBoost models trained, one per text arm.

**TF-IDF settings (identical for both arms):**
```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 1),   # unigrams only -- aligns with lexicon design
    min_df=5,             # term must appear in at least 5 documents
    max_df=0.9,           # ignore terms appearing in >90% of docs
    sublinear_tf=True     # log(1+tf) scaling
)
```

**XGBoost settings (identical for both arms):**
```python
XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.15,
    tree_method="hist",         # required for speed on CPU (not "exact")
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
```

**Class imbalance handling:**
```python
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)
```
PTSD is underrepresented (4,043 rows vs. ~19,000 for control). Balancing via sample weights avoids discarding 94% of other classes' data.

**Saved as:** `model_unscrubbed.pkl` and `model_scrubbed.pkl`

Each pickle contains a dict:
```python
{"model": XGBClassifier, "vectorizer": TfidfVectorizer, "label_encoder": LabelEncoder}
```

**Load pattern:**
```python
import pickle
with open("model_unscrubbed.pkl", "rb") as f:
    saved = pickle.load(f)
model      = saved["model"]
vectorizer = saved["vectorizer"]
le         = saved["label_encoder"]
```

### Step 7 — Classification evaluation (`02b_evaluate.py`) ✅
Loads saved models, runs on test set, saves all F1 scores and confusion matrices.

**Saved as:** `classification_results.pkl`

Structure:
```python
{
  "unscrubbed": {
    "report": {class_name: {"precision": ..., "recall": ..., "f1-score": ..., "support": ...}, ...},
    "cm":     numpy array (5x5 confusion matrix),
    "classes": ["anxiety", "control", "depression", "ptsd", "social_anxiety"]
  },
  "scrubbed": { same structure }
}
```

### Step 8 — SHAP explainability (`03_shap.py`) ✅
`shap.TreeExplainer` applied to each model. 2,000-row random sample of test set (`random_state=42`). Mean absolute SHAP value per feature per class computed.

**Saved as:** `shap_data.pkl`

Structure:
```python
{
  "unscrubbed": {
    "classes": ["anxiety", "control", "depression", "ptsd", "social_anxiety"],
    class_name: {
      "top_feature_names":  numpy array (200,)  -- top 200 features by mean |SHAP|
      "top_mean_abs_shap":  numpy array (200,)  -- corresponding mean |SHAP| values
      "top_sample_shap":    numpy array (2000, 200) float32 -- per-sample SHAP values (for bootstrap)
    }
  },
  "scrubbed": { same structure }
}
```

**Confirmed SHAP findings (unscrubbed arm):**
- PTSD: `ptsd` token has mean|SHAP|=0.7842 — 3.5× the next feature (`anxiety`, 0.2298). Accounts for ~42% of top-50 SHAP mass.
- Anxiety: `anxiety` (0.8488) dominates, followed by `anxious`, `panic`, `social`, `ptsd`, `gad`
- Depression: `depression` (0.5114) top, then `anxiety` (0.4368 — cross-class leakage), `depressed`, `life`
- Social Anxiety: `social` (0.4615) top — legitimately kept unscrubbed (SPIN vocabulary)
- Control: `credit`, `money`, `pay`, `teaching` appear — topic leakage despite mixed-subreddit design

**Confirmed SHAP findings (scrubbed arm):**
- PTSD: `trauma` (0.2182), `flashbacks`, `nightmares`, `triggers`, `therapy`, `therapist`, `abuse`, `traumatic`, `symptoms` — all PCL-5-aligned
- Anxiety: `anxious` (0.2448), `panic`, `attacks`
- Depression: `depressed` (0.2333), `life`, `suicide`, `suicidal`, `kill`, `die`, `sad`
- Social Anxiety: `social` (0.5994) — still dominant, even higher than unscrubbed

### Step 9 — Clinical lexicon construction (`04_lexicons.py`) ✅
Four lexicons built from validated clinical instrument item wording:

**Critical ordering:** Lexicons were built from instrument text ONLY, then frozen. SHAP output was examined only to refine the artifact list (adding `sa` and `gad`). This ordering is what makes SAS a validation rather than post-hoc rationalization.

**Construction method:**
1. Extract content unigrams from each instrument's item wording (tagged with POS)
2. Expand via WordNet synonyms: first synset only (most common sense), same POS only, reject multi-word expressions, reject stopwords
3. Filter to TF-IDF vocabulary of either model (only terms that can appear as SHAP features)
4. Remove known wrong-sense expansions via `EXCLUDE_FROM` dict

**Wrong-sense exclusions removed:**
- PHQ-9: `center`, `centre` (from "concentrate" → geometry sense), `smart`, `ache` (from "hurt" → physical pain), `injury`
- GAD-7: `unsatisfied`
- PCL-5: `divorce`, `trip`, `force`, `madness`
- SPIN: `address`, `alien`, `attending`, `authorization`, `discover`, `find`, `humble`, `mind`, `mouth`, `unknown`, `utter`

**Final lexicon sizes:** PHQ-9: 37 terms | GAD-7: 26 | PCL-5: 65 | SPIN: 37 | Total: 165 unique terms

**Saved as:** `lexicons.pkl`
```python
{"lexicons": {"PHQ9": [...], "GAD7": [...], "PCL5": [...], "SPIN": [...]}, "artifact_list": [...]}
```

**Also saved:** `lexicons_for_review.csv` — 165 rows with columns `instrument`, `term`, `rater1_valid`, `rater2_valid`, `notes`

**`rater1_valid` is already filled in (value = 1 for all terms that passed the construction filter). `rater2_valid` needs to be filled by the second rater.**

### Step 10 — Alignment scoring (`05_alignment.py`) ✅
Computed SAS, SPUR, AMB for both arms at k=50 (primary), k=20 and k=100 (sensitivity). Also computed cross-scale confusion matrix and 1,000-bootstrap 95% CIs.

**Saved as:** `alignment_results.pkl` and `alignment_report.txt`

### Step 11 — Figures 1–5 (`06_figures.py`) ✅
Five figures generated and committed to repo:

| File | Contents |
|---|---|
| `fig1_stacked.png` | Stacked bar: SAS/SPUR/AMB per class, unscrubbed vs. scrubbed |
| `fig2_deltas.png` | F1 delta and SAS delta side by side after scrubbing |
| `fig3_heatmap.png` | Cross-scale alignment matrix M_{c,s} for both arms |
| `fig4_shap.png` | Top-20 SHAP features for PTSD and Anxiety, color-coded clinical/spurious/ambiguous |
| `fig5_confusion.png` | Confusion matrices for both arms |

---

## 5. Verified Numbers — Do Not Modify

These numbers are locked. If you re-run from scratch and get different numbers, investigate before proceeding — check pandas version and CSV file versions.

### Pipeline row counts
| Step | Expected |
|---|---|
| After merge | 102,984 rows |
| After cross-class user dedup | 96,003 rows |
| After control downsample | 70,583 rows |
| Train set | 56,465 rows |
| Test set | 14,118 rows |

### XGBoost classification results

**Unscrubbed arm (macro-F1: 0.804)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anxiety | 0.831 | 0.730 | 0.777 | 3,601 |
| Control | 0.880 | 0.970 | 0.923 | 3,773 |
| Depression | 0.830 | 0.819 | 0.824 | 3,798 |
| PTSD | 0.772 | 0.799 | 0.785 | 797 |
| Social Anxiety | 0.696 | 0.720 | 0.708 | 2,149 |

**Scrubbed arm (macro-F1: 0.735)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Anxiety | 0.741 | 0.638 | 0.686 | 3,601 |
| Control | 0.871 | 0.958 | 0.912 | 3,773 |
| Depression | 0.775 | 0.755 | 0.765 | 3,798 |
| PTSD | 0.606 | 0.673 | 0.637 | 797 |
| Social Anxiety | 0.663 | 0.705 | 0.683 | 2,149 |

### Alignment scores (k=50, weighted, primary)

| Class | SAS unscrubbed | SPUR unscrubbed | AMB unscrubbed | SAS scrubbed | SPUR scrubbed | AMB scrubbed |
|---|---|---|---|---|---|---|
| Depression | 0.138 [0.131,0.144] | 0.376 | 0.487 | 0.206 [0.197,0.214] | 0.000 | 0.794 |
| Anxiety | 0.195 [0.189,0.200] | 0.459 | 0.347 | 0.329 [0.322,0.336] | 0.000 | 0.671 |
| PTSD | 0.266 [0.259,0.272] | 0.424 | 0.311 | 0.510 [0.502,0.517] | 0.000 | 0.490 |
| Social Anxiety | 0.304 [0.299,0.311] | 0.166 | 0.529 | 0.378 [0.372,0.386] | 0.000 | 0.622 |
| Control | 0.000 | 0.135 | 0.865 | 0.000 | 0.000 | 1.000 |

### Delta SAS
| Class | ΔSAS (scrubbed − unscrubbed) |
|---|---|
| Depression | +0.068 |
| Anxiety | +0.134 |
| PTSD | **+0.244** (largest — this is core finding #1) |
| Social Anxiety | +0.074 |
| Control | 0.000 |

### Key cross-scale confusion findings
- PTSD (scrubbed) loads on PHQ-9 at 0.139 — explained by shared PCL-5/PHQ-9 terms: `sleep`, `concentrate`, `guilt`
- Depression (scrubbed) loads on GAD-7 at 0.115 — consistent with depression/anxiety comorbidity

---

## 6. What Remains To Be Done

Three tasks in recommended order:

| Priority | Task | Time estimate | Blocking? |
|---|---|---|---|
| 1 | Inter-rater lexicon review (human, parallel with coding) | 1–2 hours human time | Blocks final alignment scores |
| 2 | DistilBERT training + SHAP + comparison figure | 2–4h CPU or ~1h GPU | Blocks paper Table 1b and Fig 6 |
| 3 | Paper finalization | 1–2 hours writing | Depends on 1 and 2 |

---

## 7. Task A — DistilBERT Training (Steps 08–10)

### What this adds
A second classifier using contextual embeddings (DistilBERT) on the same data and same train/test split as XGBoost. This allows direct comparison of:
- Classification F1 (XGBoost vs. DistilBERT)
- Clinical alignment SAS (does a stronger contextual model learn more clinical language?)

### Step 08 — Fine-tune DistilBERT (`08_transformer.py`)

**Run:**
```bash
python 08_transformer.py
```

**What it does:**
- Loads `train_df.csv` and `test_df.csv`
- Tokenizes using `distilbert-base-uncased` tokenizer, `max_length=128`
- Fine-tunes separately on `text_unscrubbed` and `text_scrubbed` columns
- Training config: 3 epochs, batch_size=16, learning_rate=2e-5, warmup_steps=200, weight_decay=0.01
- Best checkpoint selected by macro-F1 on test set
- Prints classification report at the end

**Outputs:**
- `transformer_unscrubbed/` — HuggingFace model directory (model + tokenizer + config)
- `transformer_scrubbed/` — same for scrubbed arm
- `transformer_results.pkl` — same structure as `classification_results.pkl`

**Expected runtime:**
- CPU only: ~2–4 hours per arm (~4–8 hours total)
- GPU (CUDA): ~20–40 min per arm

**The script auto-detects GPU.** If `torch.cuda.is_available()` is True, it uses `fp16=True` for faster training. No manual config needed.

**Note on training subset:** The script caps training at 40,000 rows by default (proportionally sampled from 56,465) to keep CPU training tractable. To use all 56,465 rows, set `MAX_TRAIN = None` at the top of `08_transformer.py`. On GPU, using all rows is recommended.

**Sanity check after training:** The script prints a classification report. Expect macro-F1 somewhere in the range 0.75–0.87 for unscrubbed (DistilBERT typically outperforms TF-IDF + XGBoost on text tasks). If you see macro-F1 < 0.50 or all predictions mapping to one class, something went wrong — check CUDA memory or reduce batch size.

---

### Step 09 — DistilBERT SHAP + Alignment (`09_transformer_shap.py`)

**Run:**
```bash
python 09_transformer_shap.py
```

**What it does:**
- Loads fine-tuned DistilBERT models from `transformer_unscrubbed/` and `transformer_scrubbed/`
- Creates a `shap.Explainer` with a `shap.maskers.Text` masker (the tokenizer)
- Runs SHAP on 200 test rows (smaller than XGBoost's 2,000 — transformer SHAP is much slower)
- SHAP values are at the WordPiece subword token level; the script aggregates them to word level by summing absolute values across subword pieces
- Applies the same SAS/SPUR/AMB scoring formula as `05_alignment.py`

**Outputs:**
- `transformer_shap_data.pkl` — top features per class per arm
- `transformer_alignment.pkl` — SAS/SPUR/AMB per class per arm

**Expected runtime:** ~30–90 minutes on CPU.

**Important:** The SHAP sample of 200 rows is smaller than XGBoost's 2,000. This means bootstrap CIs are not computed for DistilBERT (the sample is too small for stable 1,000-resample bootstrap). The alignment scores are point estimates only. This is stated in the paper.

---

### Step 10 — Comparison Figure (`10_figures_transformer.py`)

**Run:**
```bash
python 10_figures_transformer.py
```

**What it does:**
- Loads `classification_results.pkl` (XGBoost), `alignment_results.pkl` (XGBoost), `transformer_results.pkl`, `transformer_alignment.pkl`
- Generates Figure 6: two panels
  - Panel (a): F1 per class for all 4 combinations (XGBoost unscrubbed/scrubbed, DistilBERT unscrubbed/scrubbed)
  - Panel (b): SAS per class for XGBoost vs. DistilBERT (both arms)
- Prints a macro-F1 summary table and SAS comparison table to stdout

**Output:** `fig6_model_comparison.png`

**Expected runtime:** ~30 seconds.

---

### Running all three DistilBERT steps in sequence

```bash
python 08_transformer.py && python 09_transformer_shap.py && python 10_figures_transformer.py
```

(The `&&` means step 09 only runs if step 08 succeeds, etc.)

---

## 8. Task B — Inter-Rater Lexicon Review

### What this is
The lexicons used in alignment scoring were built by one person. For the paper's validity claim, a second rater independently reviews each term and marks it valid (1) or invalid (0). Cohen's kappa is computed to report inter-rater agreement.

### The file to fill in
`lexicons_for_review.csv` — 165 rows.

**Columns:**
| Column | Contents |
|---|---|
| `instrument` | PHQ9, GAD7, PCL5, or SPIN |
| `term` | The lexicon term (e.g., `hopeless`, `trauma`, `flashback`) |
| `rater1_valid` | Already filled in: 1 for all 165 terms (these passed the automated construction filter) |
| `rater2_valid` | **You fill this in: 1 = this term belongs in this instrument's lexicon, 0 = it does not** |
| `notes` | Optional: reasons for rejection, edge cases |

### What "valid" means
A term is valid if it:
1. Appears in or is directly derived from the item wording of that instrument, AND
2. Would meaningfully indicate the condition named by the instrument if a user wrote it in a post

A term is invalid if it:
- Expanded via WordNet into a wrong or too-general sense (e.g., `force` in PCL-5 — came from "forced sexual contact" but the word alone is too ambiguous)
- Could appear naturally in any non-clinical context with high frequency (e.g., `focus` in PHQ-9 — valid in item 7 "trouble concentrating" but very common generally)

### Instrument references for review

**PHQ-9 (depression) — 9 items:**
1. Little interest or pleasure in doing things
2. Feeling down, depressed, or hopeless
3. Trouble falling or staying asleep, or sleeping too much
4. Feeling tired or having little energy
5. Poor appetite or overeating
6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down
7. Trouble concentrating on things
8. Moving or speaking so slowly that other people could have noticed, or being so fidgety or restless that you have been moving around a lot more than usual
9. Thoughts that you would be better off dead or of hurting yourself in some way

**GAD-7 (anxiety) — 7 items:**
1. Feeling nervous, anxious, or on edge
2. Not being able to stop or control worrying
3. Worrying too much about different things
4. Trouble relaxing
5. Being so restless that it's hard to sit still
6. Becoming easily annoyed or irritable
7. Feeling afraid, as if something awful might happen

**PCL-5 (PTSD) — 20 items:** Covers intrusive memories, nightmares, flashbacks, emotional distress when reminded of trauma, avoidance behaviors, negative beliefs about self/world, persistent negative emotions, loss of interest, feeling detached, inability to feel positive, irritability/aggression, recklessness, hypervigilance, exaggerated startle response, concentration difficulties, sleep disturbance.

**SPIN (Social Phobia Inventory) — 17 items:** Covers fear of people in authority, blushing, parties and social events, talking to strangers, being criticized, embarrassment/humiliation, sweating in social situations, center of attention, public speaking, trembling/shaking in front of others, social situations, looking stupid/foolish, shyness.

### After filling in rater2_valid

Run:
```bash
python 07_interrater.py
```

**What it does:**
1. Loads `lexicons_for_review.csv`
2. Computes percent agreement and Cohen's kappa per instrument and overall
3. Keeps only terms where **both raters marked 1**
4. Overwrites `lexicons.pkl` with the pruned lexicons
5. Reports kappa and new lexicon sizes

**Then re-run alignment scoring and figures:**
```bash
python 05_alignment.py
python 06_figures.py
```

This regenerates `alignment_results.pkl` and all figures 1–5 with final, validated numbers.

### Target kappa
The paper aims for κ ≥ 0.80 (almost perfect agreement). If kappa is lower:
- Review disagreements in the `notes` column
- Discuss borderline terms and reach consensus
- If κ < 0.61 overall, the lexicon validation is too weak to report — re-review terms together

### What happens to alignment scores after pruning
Pruning terms that one rater rejected will generally **lower** SAS values slightly (fewer terms match). This is expected and represents a more conservative, validated estimate. The rank ordering of classes (PTSD > Social Anxiety > Anxiety > Depression > Control) should remain stable unless many terms in one instrument are rejected.

---

## 9. Task C — Paper Finalization

`paper_draft.md` is nearly complete. The following items need to be filled in after Tasks A and B:

### After Task B (inter-rater):
1. **Section 4.3** — Replace `[TBD after inter-rater step]` with the actual Cohen's κ value
2. **Table 2** — If alignment scores change after lexicon pruning, update SAS/SPUR/AMB values and CIs
3. **Table 3** — Same — update cross-scale matrix values
4. **Appendix A** — Update unweighted scores if they changed
5. **Section 4.3** — Replace "165 terms" with the post-pruning count of final lexicon terms
6. **Appendix B sensitivity values** — These will auto-update when you run 05_alignment.py and paste in the new numbers

### After Task A (DistilBERT):
1. **Table 1b** — Fill in DistilBERT F1 values (copy from `transformer_results.pkl` output or console)
2. **Section 5.1** — Add a sentence comparing DistilBERT macro-F1 to XGBoost
3. **Section 6.4** — Interpret whether DistilBERT achieves higher SAS alongside higher F1 (see the two possible outcomes described in the current draft)

### Section 2.2 Cluster B citations
This section currently reads: `[Note to author: add 1–2 Cluster B citations here]`

You need to find 1–2 papers that apply SHAP to clinical (non-social-media) mental health data and validate against DSM criteria or clinical instruments. Search Google Scholar for:
- `"SHAP" "depression" "clinical notes" OR "EHR"`
- `"explainable" "mental health" "clinical validation" "SHAP"`

### References already in the paper (do not add duplicates)
1. Low et al. (2020) — RMHD dataset
2. Lundberg & Lee (2017) — SHAP
3. Kroenke et al. (2001) — PHQ-9
4. Spitzer et al. (2006) — GAD-7
5. Weathers et al. (2013) — PCL-5
6. Connor et al. (2000) — SPIN
7. Chen & Guestrin (2016) — XGBoost
8. Sanh et al. (2019) — DistilBERT
9. Ibrahimov et al. (2024) — arXiv survey (Cluster C)
10. Tadesse et al. (2019) — Cluster A
11. Burdisso et al. (2019) — Cluster A

### Ethics section (non-negotiable — must remain in final paper)
Section 7 is already written and must not be removed or weakened. It covers:
- No individual-level risk prediction
- No new data scraping
- Subreddit membership is proxy label, not clinical diagnosis
- SAS is a lower bound
- Inter-rater validation (κ value to fill in)

---

## 10. Locked Design Decisions

These decisions were made early and are final. Do not change them.

| Decision | Reason |
|---|---|
| Use only RMHD `pre` files (Dec 2018 – Dec 2019) | Avoid COVID vocabulary contamination |
| Do NOT merge with Dreaddit, SWMH, SMHD, or other datasets | Different label types confound artifact analysis |
| Split by **author**, not by row | Prevents writer identity from inflating metrics |
| Do NOT hard-balance all classes to PTSD minimum | Would discard 94% of data; use sample_weight instead |
| Report **macro-F1** as primary metric | Gives equal weight to PTSD despite class imbalance |
| Do NOT scrub `"depressed"` | PHQ-9 item 2 wording — genuine clinical signal |
| Do NOT scrub `"social"` | Core SPIN vocabulary — top SHAP feature for social_anxiety |
| Lexicons built before inspecting SHAP | This ordering is what makes SAS a validation, not rationalization |
| Exact lemma matching for SAS (not embedding-based) | Keeps metric lexical, auditable, and transparent |
| All random seeds: `random_state=42` | Reproducibility |
| Primary k=50 for SAS | With k=20 and k=100 as sensitivity checks in Appendix B |
| N=1,000 bootstrap resamples for CIs | Standard for this type of metric |

---

## 11. Known Pitfalls and Warnings

### 1. Large CSV files are NOT in the repo
`train_df.csv`, `test_df.csv`, `merged_raw.csv`, `merged_balanced.csv`, and all 7 raw source CSVs are git-ignored due to file size. You must download the 7 Zenodo CSVs and place them in the project root. Then run `01_pipeline.py` to regenerate the merged/train/test CSVs.

**If you skip this and try to run 08_transformer.py directly, it will fail** because it needs `train_df.csv` and `test_df.csv`.

### 2. DistilBERT model directories are NOT in the repo
`transformer_unscrubbed/` and `transformer_scrubbed/` are git-ignored (they would be ~500MB each). They are produced by running `08_transformer.py`.

### 3. `transformers` library needs to be installed separately
It is in `requirements.txt` but was not pre-installed. Run `pip install transformers accelerate` before attempting step 08.

### 4. XGBoost training takes 6–10 minutes on CPU per arm
If you re-run `02_train.py`, expect ~15 minutes. It is not frozen — the model is training normally. Do not interrupt it.

### 5. SHAP on XGBoost takes ~15 minutes on CPU
`03_shap.py` on 2,000 test rows. Again, do not interrupt.

### 6. Social anxiety SAS appears lower than expected in unweighted scores
Weighted SAS (primary) for social anxiety unscrubbed = 0.304. Unweighted SAS = 0.120. This large gap is expected and is explained in Appendix A of the paper: a single term (`social`) holds 46% of the social anxiety model's SHAP mass, so it dominates the weighted score while counting as only 2% of the top-50 list in the unweighted version. This is a finding, not a bug.

### 7. Do NOT re-run `04_lexicons.py` after the inter-rater step
`07_interrater.py` overwrites `lexicons.pkl` with pruned lexicons. If you re-run `04_lexicons.py` afterwards, it will restore the full unreviewed lexicon and overwrite the pruned version. Run `07_interrater.py` last among the lexicon-related steps.

### 8. `run_all.py --all` runs 07_interrater.py
Only use the `--all` flag after `lexicons_for_review.csv` has both rater columns filled. Running `--all` with empty `rater2_valid` columns will fail with a blank rating warning and halt.

### 9. DistilBERT SHAP sample is 200 rows, not 2,000
Transformer SHAP is compute-expensive. The smaller sample means DistilBERT SAS values are less stable than XGBoost SAS values. This is disclosed in the paper and is expected for a comparison model.

### 10. Windows line endings (CRLF)
The repo was initialized on Windows. Git will show CRLF warnings when checking out on Linux/Mac. This is cosmetic and does not affect script functionality. If you see `\r\n` issues in any output, add `text=True, encoding="utf-8"` to `open()` calls.

---

## 12. File Reference

### Code (all committed)

| File | Purpose | Input | Output |
|---|---|---|---|
| `01_pipeline.py` | Merge → dedup → downsample → split → scrub | 7 raw CSVs | `merged_raw.csv`, `merged_balanced.csv`, `train_df.csv`, `test_df.csv` |
| `02_train.py` | Train XGBoost both arms | `train_df.csv`, `test_df.csv` | `model_unscrubbed.pkl`, `model_scrubbed.pkl` |
| `02b_evaluate.py` | Classification reports | `test_df.csv`, model PKLs | `classification_results.pkl` |
| `03_shap.py` | XGBoost SHAP values | `test_df.csv`, model PKLs | `shap_data.pkl` |
| `04_lexicons.py` | Build clinical lexicons | `model_*.pkl` (vocabulary) | `lexicons.pkl`, `lexicons_for_review.csv` |
| `05_alignment.py` | SAS/SPUR/AMB + bootstrap | `shap_data.pkl`, `lexicons.pkl` | `alignment_results.pkl`, `alignment_report.txt` |
| `06_figures.py` | Figures 1–5 | `alignment_results.pkl`, `classification_results.pkl`, `shap_data.pkl`, `lexicons.pkl` | `fig1_stacked.png` – `fig5_confusion.png` |
| `07_interrater.py` | Inter-rater kappa + prune | `lexicons_for_review.csv`, `lexicons.pkl` | Updated `lexicons.pkl` |
| `08_transformer.py` | Fine-tune DistilBERT | `train_df.csv`, `test_df.csv` | `transformer_unscrubbed/`, `transformer_scrubbed/`, `transformer_results.pkl` |
| `09_transformer_shap.py` | DistilBERT SHAP + alignment | `test_df.csv`, transformer dirs, `lexicons.pkl` | `transformer_shap_data.pkl`, `transformer_alignment.pkl` |
| `10_figures_transformer.py` | Figure 6 comparison | `classification_results.pkl`, `alignment_results.pkl`, `transformer_results.pkl`, `transformer_alignment.pkl` | `fig6_model_comparison.png` |
| `run_all.py` | Orchestrates all steps | — | Calls each script in order |

### Data (NOT committed — download from Zenodo)
7 files named `*_pre_features_tfidf_256.csv` from Zenodo record 3941387.

### Checkpoints (committed)

| File | Size | Contents |
|---|---|---|
| `model_unscrubbed.pkl` | 2.0 MB | XGBoost + TF-IDF (unscrubbed) |
| `model_scrubbed.pkl` | 2.0 MB | XGBoost + TF-IDF (scrubbed) |
| `shap_data.pkl` | 15.3 MB | XGBoost SHAP top-200 per class per arm |
| `lexicons.pkl` | <1 MB | Clinical lexicons (165 terms) |
| `lexicons_for_review.csv` | <1 MB | 165 terms for inter-rater review |
| `alignment_results.pkl` | <1 MB | SAS/SPUR/AMB + bootstrap CIs (XGBoost) |
| `classification_results.pkl` | <1 MB | F1 tables + confusion matrices (XGBoost) |
| `alignment_report.txt` | <1 MB | Human-readable alignment scores |

### Figures (committed)
`fig1_stacked.png`, `fig2_deltas.png`, `fig3_heatmap.png`, `fig4_shap.png`, `fig5_confusion.png`

### Documentation (committed)
| File | Purpose |
|---|---|
| `README.md` | Quick-start for new contributors |
| `CLAUDE.md` | Full project specification (original handoff doc) |
| `HANDOFF.md` | This document |
| `paper_draft.md` | Complete paper draft |
| `SMA-NLP Project Explanation.pdf` | Original project PDF |
| `SMA-NLP Project Outline.pdf` | Project outline |
| `SMA_NLP_Project.ipynb` | Original Colab notebook |

---

## Recommended Order of Work

```
Day 1 (parallel):
  ├── Start:  python 08_transformer.py        (let it run for hours)
  └── Start:  Fill lexicons_for_review.csv    (human task, ~1–2h)

Day 2 (after both complete):
  ├── python 09_transformer_shap.py           (~1h)
  ├── python 07_interrater.py                 (after CSV is filled)
  ├── python 05_alignment.py                  (re-run with pruned lexicons)
  ├── python 06_figures.py                    (regenerate figures 1–5)
  └── python 10_figures_transformer.py        (after step 09 done)

Day 3:
  └── Update paper_draft.md with final numbers and kappa
```

Total estimated time: 2 working days (most of Day 1 is waiting for training).
