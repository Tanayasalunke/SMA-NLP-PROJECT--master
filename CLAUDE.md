# SMA-NLP Project — CLAUDE.md

## Project Title
**Decoding Mental Health States from Social Media Text: A Multi-Class Classification Approach with Clinical Symptom-Alignment Validation**

## Core Thesis
Build a multi-class classifier for mental health states from Reddit posts, then use SHAP explainability scored against real clinical instruments (PHQ-9, GAD-7, PCL-5, SPIN) to quantify how much of the model's decision-making reflects clinically valid symptom language versus spurious dataset artifacts (subreddit self-references, jargon).

---

## Repository Layout

```
SMA-NLP Project/
├── CLAUDE.md                                  ← this file
├── SMA_NLP_Project.ipynb                      ← main notebook (Google Colab origin)
├── SMA-NLP Project Explanation.pdf/.docx      ← project handoff document
├── SMA-NLP Project Outline.pdf
│
│── Raw source CSVs (all 7 from RMHD Zenodo record 3941387)
│   ├── depression_pre_features_tfidf_256.csv   → label: depression   (21,209 rows)
│   ├── anxiety_pre_features_tfidf_256.csv      → label: anxiety      (19,976 rows)
│   ├── ptsd_pre_features_tfidf_256.csv         → label: ptsd         (4,483 rows)
│   ├── socialanxiety_pre_features_tfidf_256.csv→ label: social_anxiety(12,277 rows)
│   ├── jokes_pre_features_tfidf_256.csv        → label: control      (19,249 rows)
│   ├── personalfinance_pre_features_tfidf_256.csv→label: control     (23,440 rows)
│   └── teaching_pre_features_tfidf_256.csv     → label: control      (2,350 rows)
│
│── Checkpoint files (originally in Google Drive; download locally to work)
│   ├── merged_raw.csv          (102,984 rows — Step 1 output)
│   ├── merged_balanced.csv     (70,583 rows — Step 3 output)
│   ├── train_df.csv            (56,465 rows — Step 4+5 output)
│   ├── test_df.csv             (14,118 rows — Step 4+5 output)
│   ├── model_unscrubbed.pkl    (XGBoost + TF-IDF, unscrubbed arm)
│   ├── model_scrubbed.pkl      (XGBoost + TF-IDF, scrubbed arm)
│   └── shap_top_features.pkl   (NOT YET CREATED — Task 1)
```

---

## Dataset

- **Source:** Reddit Mental Health Dataset (RMHD), Zenodo record 3941387
- **Timeframe:** `pre` files only (Dec 2018–Dec 2019, pre-pandemic — avoids COVID vocabulary contamination)
- **Columns kept per file:** `["subreddit", "author", "date", "post"]` — the 250+ precomputed TF-IDF/LIWC columns are dropped; we recompute our own features for reproducibility
- **5 classes:** `depression`, `anxiety`, `ptsd`, `social_anxiety`, `control`
- **Control design:** 3 mixed subreddits (jokes + personalfinance + teaching) — reduces topic-vocabulary leakage vs. single-subreddit control

### Locked design decisions
- Do NOT merge with Dreaddit, SWMH, SMHD, or any other external dataset (different label types / collection pipelines confound the spurious-feature analysis)
- Do NOT hard-balance all classes to PTSD's size — use `sample_weight` / class weights instead; report macro-F1 as the primary metric
- All random seeds: `random_state=42` throughout

---

## Data Pipeline — Fully Executed (Verified Numbers)

### Step 1 — Merge
```python
pd.read_csv(f, usecols=lambda c: c in KEEP, engine="c", on_bad_lines="skip")
```
- **Result:** 102,984 rows → saved as `merged_raw.csv`
- Use `engine="c"` — `engine="python"` was ruled out (same row count but slower; earlier confusion was due to incomplete Colab uploads, not the parser)

### Step 2 — Deduplicate cross-class users
```python
overlap = data.groupby("author")["label"].nunique()
leaky_users = overlap[overlap > 1]
data = data[~data["author"].isin(leaky_users.index)]
```
- 3,359 users removed → **96,003 rows**

### Step 3 — Downsample control class
- Target: 25,000 control rows, sampled proportionally across the 3 control subreddits
- **Result:** 70,583 rows → saved as `merged_balanced.csv`
  - depression: 19,210 | control: 18,941 | anxiety: 17,701 | social_anxiety: 10,688 | ptsd: 4,043

### Step 4 — Author-level train/test split
```python
train_authors, test_authors = train_test_split(authors, test_size=0.2, random_state=42)
```
- Split by unique `author` (not by row) — prevents model from learning writer identity
- **Train: 56,465** | **Test: 14,118**

### Step 5 — Scrubbed / Unscrubbed text arms
Two parallel columns created on both train and test:
- `text_unscrubbed` = raw `post` column
- `text_scrubbed` = post with artifact terms removed via regex

**Artifact terms (plain substring removal):**
```python
ARTIFACT_TERMS = [
    "depression", "anxiety", "ptsd", "social anxiety", "socialanxiety",
    "diagnosed", "diagnosis", "dae", "throwaway", "tw:", "trigger warning",
    "r/depression", "r/anxiety", "r/ptsd", "r/socialanxiety",
]
```

**Word-boundary-matched terms (`\bterm\b`):**
```python
ARTIFACT_TERMS_WB = ["sa", "gad"]
```
These were added after SHAP revealed them as un-scrubbed self-reference shorthand leaking into top features.

**Deliberately NOT scrubbed:**
- `"depressed"` — mirrors PHQ-9 item wording ("feeling down, depressed, or hopeless")
- `"social"` — core SPIN (Social Phobia Inventory) vocabulary

Saved as: `train_df.csv`, `test_df.csv`

---

## Models — Trained and Saved

### TF-IDF configuration (identical for both arms)
```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 1),   # unigrams only
    min_df=5,
    max_df=0.9,
    sublinear_tf=True
)
```

### XGBoost configuration (identical for both arms)
```python
XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.15,
    tree_method="hist",           # critical for speed — "exact" is very slow
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
```
Class imbalance: `compute_sample_weight(class_weight="balanced", y=y_train)` passed as `sample_weight` to `.fit()`.

### Results — Unscrubbed arm (macro-F1: 0.805, accuracy: 0.822)
| Class          | Precision | Recall | F1    | Support |
|----------------|-----------|--------|-------|---------|
| anxiety        | 0.832     | 0.734  | 0.780 | 3601    |
| control        | 0.881     | 0.969  | 0.923 | 3773    |
| depression     | 0.831     | 0.821  | 0.826 | 3798    |
| ptsd           | 0.777     | 0.801  | 0.789 | 797     |
| social_anxiety | 0.697     | 0.720  | 0.708 | 2149    |

### Results — Scrubbed arm (macro-F1: 0.735, accuracy: 0.766)
| Class          | Precision | Recall | F1    | Support |
|----------------|-----------|--------|-------|---------|
| anxiety        | 0.740     | 0.634  | 0.683 | 3601    |
| control        | 0.871     | 0.958  | 0.912 | 3773    |
| depression     | 0.778     | 0.751  | 0.764 | 3798    |
| ptsd           | 0.596     | 0.675  | 0.633 | 797     |
| social_anxiety | 0.656     | 0.709  | 0.682 | 2149    |

### Headline scrubbing delta (core paper finding #1)
| Class          | Unscrubbed F1 | Scrubbed F1 | Δ      |
|----------------|---------------|-------------|--------|
| Anxiety        | 0.780         | 0.683       | −0.097 |
| Control        | 0.923         | 0.912       | −0.011 |
| Depression     | 0.826         | 0.764       | −0.062 |
| **PTSD**       | **0.789**     | **0.633**   | **−0.156** |
| Social Anxiety | 0.708         | 0.682       | −0.026 |
| **Macro-F1**   | **0.805**     | **0.735**   | **−0.070** |

**Interpretation:** PTSD's large degradation = model relied heavily on literal token "ptsd" (confirmed by SHAP). Social anxiety barely moved → its lower ceiling reflects genuine class confusability with anxiety, not artifact-dependence.

### Headline cross-scale confusion finding (core paper finding #2)
- Anxiety ↔ Social Anxiety are each other's largest error in both arms (clinically expected — GAD-7/SPIN overlap)
- After scrubbing, this confusion gets **worse**: anxiety→social_anxiety rises 408→434, anxiety→depression rises 326→488
- Framing: scrubbing selectively degrades boundaries between clinically adjacent classes more than between classes with distinct vocabulary

### Model reload pattern
```python
import pickle
with open(base_path + "model_unscrubbed.pkl", "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
vectorizer = saved["vectorizer"]
le = saved["label_encoder"]
```

---

## SHAP Explainability (Run Once — Needs Re-run and Save)

**Method:** `shap.TreeExplainer` on XGBoost model, on a 2,000-row random sample (`random_state=42`) of the test set. Mean absolute SHAP value per feature per class, top-50 extracted per class.

**Status:** Run once on an earlier session's models (deterministic, same numbers). Currently needs to be re-run on freshly saved `.pkl` files and saved to `shap_top_features.pkl`.

### Confirmed SHAP findings

**PTSD (most important finding):**
- Unscrubbed top: `ptsd` (|SHAP|=0.7842) — 3.5× the next feature (`anxiety`, 0.2298)
- Scrubbed top: `trauma` (0.2182), then `flashbacks`, `nightmares`, `triggers`, `therapy`, `therapist`, `abuse`, `traumatic`, `symptoms` — all PCL-5-aligned

**Social Anxiety:**
- Top feature both arms: `social` (0.4615 unscrubbed → 0.5994 scrubbed)
- Deliberately kept — legitimate SPIN vocabulary, not artifact
- `sa` appeared in unscrubbed top-20 (0.0645) → triggered adding "sa" to artifact list

**Anxiety:**
- Unscrubbed top: `anxiety` (0.8488), then `anxious`, `panic`, `social`, `ptsd`, `gad`
- `gad` appearance → triggered adding "gad" to artifact list
- Scrubbed top: `anxious` (0.2448), `panic`, `attack(s)`

**Depression:**
- Unscrubbed top: `depression` (0.5114), then `anxiety` (0.4368 — cross-class leakage), `depressed`, `life`
- Scrubbed top: `depressed` (0.2333), `life`, `suicide`, `suicidal`, `kill`, `die`, `sad`

**Control:**
- Topic leakage despite mixed design: `credit`, `money`, `pay`, `teaching` appear
- State as limitation: mixing subreddits *reduced* but did not *eliminate* topic-vocabulary signal

---

## Alignment Scoring Framework (Defined — Not Yet Coded)

This is the paper's core methodological contribution.

### Setup
- `F_c^(k)` = top-k SHAP features for class c. **Primary k=50**, sensitivity at k ∈ {20, 50, 100}
- `L_c` = clinical lexicon per class: PHQ-9 (depression), GAD-7 (anxiety), PCL-5 (ptsd), SPIN (social_anxiety). Lemmatized unigrams. Control has no `L_c`.
- `A` = artifact list (ARTIFACT_TERMS + ARTIFACT_TERMS_WB, including `sa` and `gad` explicitly)
- `φ_c(f)` = mean absolute SHAP value for feature f in class c
- `m(f, S) = 1` if `lemma(f) ∈ S`, else 0

### Three scores (weighted variant — PRIMARY)
```
SAS_c  = Σ_{f∈F_c^k} φ_c(f)·m(f,L_c)  / Σ_{f∈F_c^k} φ_c(f)
SPUR_c = Σ_{f∈F_c^k} φ_c(f)·m(f,A)·(1-m(f,L_c)) / Σ_{f∈F_c^k} φ_c(f)
AMB_c  = 1 - SAS_c - SPUR_c
```
Also compute unweighted variant (simple proportion) for appendix.

### Cross-scale confusion matrix (R4)
```
M_{c,s} = Σ_{f∈F_c^k} φ_c(f)·m(f,L_s) / Σ_{f∈F_c^k} φ_c(f)
```
for s ∈ {PHQ-9, GAD-7, PCL-5, SPIN}. Diagonal = SAS_c.

### Scrubbed-vs-unscrubbed delta (R3)
```
ΔSAS_c = SAS_c^(scrubbed) - SAS_c^(unscrubbed)
```

### Parameter decisions (finalized)
- Matching = exact lemma-level match (spaCy or WordNet lemmatizer). NOT embedding-based for primary score.
- **Lexicons must be frozen before inspecting SHAP output** (state this in paper as what makes it a validation, not rationalization)
- Multi-word instrument phrases decomposed into content unigrams (consistent with `ngram_range=(1,1)`)
- Bootstrap 1,000 resamples of test set → 95% CIs on all scores
- Get second rater to independently validate lexicon terms; report Cohen's kappa

---

## Remaining Tasks (In Order)

### Task 1 — SHAP re-run and save (IMMEDIATE)
Re-run SHAP on saved `.pkl` models. Save output:
```python
import pickle
with open(base_path + "shap_top_features.pkl", "wb") as f:
    pickle.dump({"unscrubbed": top_features_unscrubbed, "scrubbed": top_features_scrubbed}, f)
```

### Task 2 — Build 4 clinical lexicons + finalize artifact list (Week 5)
- Extract content words from PHQ-9, GAD-7, PCL-5, SPIN item wording
- Expand via WordNet synonyms + manual pass
- Lemmatize both lexicons and SHAP feature names before matching
- Add `sa`, `gad` explicitly to artifact list A
- Second rater independently validates lexicon terms → compute agreement

### Task 3 — Implement alignment scoring code (Week 6)
- SAS_c, SPUR_c, AMB_c (weighted primary + unweighted appendix)
- Cross-scale confusion matrix M_{c,s}
- ΔSAS_c per class
- Bootstrap 1,000 resamples for 95% CIs
- Sensitivity check at k ∈ {20, 50, 100}

### Task 4 — Four core figures
1. Stacked bar chart: SAS/SPUR/AMB per class (sums to 1)
2. Scrubbed-vs-unscrubbed deltas: accuracy delta + alignment delta, side by side
3. Class × instrument alignment heatmap (M_{c,s} matrix)
4. Annotated SHAP plot with features color-coded clinical / spurious / ambiguous

### Task 5 — Decision: DistilBERT/RoBERTa comparison model
Not yet built. Options: (a) build now for fuller model comparison, or (b) proceed XGBoost-only and cite transformer comparison as future work. The alignment framework (Task 3) does NOT require a transformer — XGBoost + TreeExplainer is sufficient for the core contribution.

### Task 6 — Write-up (Weeks 7–8)
Paper structure:
1. Introduction — problem, thesis, 3 contributions
2. Related Work — Cluster A/B/C framing → "intersection is empty; we fill it"
3. Data — condensed Sections 2–3
4. Methods — 4a classifiers, 4b SHAP setup, 4c alignment framework (longest)
5. Results — R1 (classification tables), R2 (per-class SAS/SPUR/AMB), R3 (scrubbed/unscrubbed deltas — accuracy AND alignment), R4 (confusion matrix + cross-scale heatmap)
6. Discussion — implications for prior RMHD benchmarks; anxiety/social-anxiety finding; recommend future work report alignment alongside F1
7. Limitations & Ethics — see ethics section below

---

## Ethics & Limitations (Non-Negotiable — Must Appear in Paper)

- No individual-level risk prediction or diagnosis — research/validation study only
- No new data scraping or self-labeling of real users — only existing public RMHD dataset
- Subreddit membership is a weak/proxy label, not a clinical diagnosis — state explicitly
- RMHD collected via Pushshift API; re-use subject to Reddit API terms (one sentence)
- Control class: mixing subreddits reduced but did not eliminate topic-vocabulary leakage
- SAS is a lexical, lower-bound measure — cannot credit the model for symptom meaning expressed without a lexicon keyword. Frame as conservative, auditable floor.

---

## Environment & Reproducibility

### Running the notebook
- Originally developed in **Google Colab** with Drive mounted at `/content/drive/MyDrive/SMA-NLP Project/`
- For local VS Code: update `base_path` to the local directory (e.g., the folder containing this CLAUDE.md)
- Checkpoint CSVs and `.pkl` files need to be downloaded from Drive to work locally

### Sanity check after environment migration
Re-run Step 1 merge and confirm exactly **102,984 rows** with the per-class breakdown shown. Any difference → investigate pandas version or file version before proceeding.

### Known slow step
XGBoost training: ~6–10 minutes per arm with `tree_method="hist"` on CPU. Do not assume it's broken during this time.

### Known pitfall (solved)
`engine="python"` in `pd.read_csv` is slower and was initially suspected of dropping rows — it was actually incomplete Colab uploads causing issues. Use `engine="c"` with `on_bad_lines="skip"`.

### Dependencies
```
pandas, scikit-learn, xgboost, shap, re, pickle
spacy or nltk (for lemmatization in Tasks 2–3)
matplotlib / seaborn (for Task 4 figures)
```
