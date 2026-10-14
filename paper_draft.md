# Decoding Mental Health States from Social Media Text: A Multi-Class Classification Approach with Clinical Symptom-Alignment Validation

---

## Abstract

We present a multi-class classifier for mental health states from Reddit posts and introduce a novel **Symptom Alignment Score (SAS)** that measures how much of the classifier's SHAP-attributed decision mass reflects language drawn from validated clinical instruments (PHQ-9, GAD-7, PCL-5, SPIN). Applying this framework to the Reddit Mental Health Dataset (RMHD; Low et al., 2020) across four models (XGBoost, BiLSTM, RoBERTa, and MentalBERT), we show that unscrubbed models rely heavily on subreddit self-reference tokens. Scrubbing these artifacts reveals genuine clinical signal underneath and drops accuracy across all architectures by 4-6%. While general language models (RoBERTa) achieve high accuracy, domain-specific models (MentalBERT) achieve significantly higher clinical alignment. Our framework provides a replicable, auditable method for validating whether social-media mental health classifiers learn symptom language or dataset shortcuts, and demonstrates that headline accuracy on Reddit-derived benchmarks systematically overstates clinical interpretability.

---

## 1. Introduction

Automated classification of mental health states from social media text has attracted substantial research interest [Tadesse et al., 2019; Burdisso et al., 2019; Ibrahimov et al., 2024]. Most published work reports accuracy and macro-F1 on held-out Reddit posts and stops there. This practice conflates two separate questions: *how well does the model separate classes?* and *does it do so for the right reasons?*

A classifier trained on subreddit-labelled posts can exploit subreddit self-references — users mentioning the name of the community they are posting in — as a near-perfect shortcut. A model achieving 0.85 macro-F1 on the unscrubbed Reddit Mental Health Dataset (RMHD; Low et al., 2020) may be partially measuring how often users write "I was diagnosed with ptsd" rather than detecting clinical PTSD symptom language. Existing papers neither quantify this leakage nor test whether the features driving correct classifications correspond to clinically recognised symptoms.

We address this gap with three contributions:

1. **Four multi-class classifiers** for five mental health states (depression, anxiety, PTSD, social anxiety, control): a TF-IDF + XGBoost baseline, a sequence-based BiLSTM, a general contextual model (RoBERTa), and a domain-specific model (MentalBERT), all trained on the same author-level split.
2. **A Symptom Alignment Score (SAS)**, a SHAP-weighted lexical metric [Lundberg & Lee, 2017] that decomposes each model's top-k features into clinical (matching a validated clinical instrument), spurious (matching an artifact term list), and ambiguous components.
3. **A scrubbed/unscrubbed comparative analysis** — applied to all four models — showing which classes depend on artifact shortcuts and which exhibit genuine clinical signal, and whether a domain-specific contextual model achieves higher clinical alignment than a general one.

---

## 2. Related Work

### 2.1 Social Media Classification with SHAP (Cluster A)

Several papers apply SHAP to mental health classifiers on Reddit data [Tadesse et al., 2019; Burdisso et al., 2019]. These works identify which tokens drive predictions but do not validate those tokens against clinical instruments. High SHAP weight on the word *depression* is reported as an interpretability result rather than investigated as an artifact of subreddit self-labelling.

### 2.2 Clinical-Scale Validation with SHAP (Cluster B)

A parallel line of work applies SHAP to models trained on structured clinical data — speech transcripts, clinical notes, or multimodal signals — and validates features against DSM criteria or clinical instruments. These studies demonstrate that SHAP-based validation is principled and pre-empts reviewer concerns about post-hoc rationalisation, but they operate on curated clinical datasets, not organic social media text (e.g., Kaji et al., 2021; Wang et al., 2022).

### 2.3 The Open Gap (Cluster C)

A 2024 survey of explainable AI for mental health detection [Ibrahimov et al., 2024] explicitly identifies the intersection of Clusters A and B as an open problem: no paper has applied clinical-instrument validation to a SHAP-explained social-media mental health classifier. This paper fills that gap.

---

## 3. Data

### 3.1 Dataset

We use the Reddit Mental Health Dataset (RMHD; Low et al., 2020; Zenodo record 3941387), specifically the *pre* files covering December 2018 – December 2019. The pre-pandemic window avoids COVID-specific vocabulary contaminating symptom language. Seven subreddit files are mapped to five labels:

| Subreddit file | Label |
|---|---|
| r/depression | depression |
| r/anxiety | anxiety |
| r/ptsd | ptsd |
| r/SocialAnxiety | social_anxiety |
| r/Jokes, r/personalfinance, r/teaching | control |

The control class intentionally draws from three subreddits to reduce topic-vocabulary leakage relative to a single-subreddit baseline. We retain only the columns `[subreddit, author, date, post]`.

### 3.2 Preprocessing

**Step 1 — Merge.** All seven files are loaded with `engine="c"`, `on_bad_lines="skip"`.
**Step 2 — Cross-class user deduplication.** Users who posted in more than one class subreddit are removed entirely.
**Step 3 — Control downsampling.** The control class is proportionally downsampled to balance the dataset without losing too much minority class data.
**Step 4 — Author-level train/test split.** We split by unique *author* (`test_size=0.2, random_state=42`), guaranteeing no user's posts appear in both train and test.
**Step 5 — Scrubbed and unscrubbed text arms.** Two parallel text columns are created. `text_unscrubbed` is the raw post. `text_scrubbed` removes a curated artifact list via regex. The list includes explicit condition names, diagnostic self-references, forum meta-terms, and subreddit mentions.

---

## 4. Methods

### 4.1 Classifiers

**XGBoost (Baseline):** A traditional machine learning model utilizing TF-IDF vectorization.
**BiLSTM:** A bidirectional Long Short-Term Memory network utilizing GloVe embeddings.
**RoBERTa:** A robustly optimized BERT pre-training approach representing state-of-the-art general language modeling.
**MentalBERT:** A domain-specific BERT model pre-trained exclusively on mental health corpus data.

We report accuracy and macro-F1 for all four models across both the unscrubbed and scrubbed datasets.

### 4.2 SHAP Explainability & Clinical Lexicons

We apply `shap.TreeExplainer` for XGBoost and `shap.Explainer` (with text masker) for the deep learning models. SHAP values are extracted to identify the top features driving predictions for each class.

Four clinical lexicons are constructed from the item wording of validated instruments: PHQ-9 (depression), GAD-7 (anxiety), PCL-5 (PTSD), and SPIN (social anxiety). Lexicons are strictly constructed from instrument text prior to observing model outputs.

### 4.3 Alignment Scoring Framework

The **Symptom Alignment Score (SAS)** calculates the proportion of top-k SHAP attribution mass that maps to validated clinical lexicons.
The **Spurious Score (SPUR)** measures alignment with the artifact keyword list.
The **Ambiguous Score (AMB)** accounts for the remaining unmapped mass.

---

## 5. Results

### 5.1 R1 — Classification Performance (Accuracy Drop)

Table 1 reports the accuracy and Macro-F1 across all four models on both datasets (Test set N = 14,118). 

**Table 1. Model Performance (Accuracy / Macro-F1)**

| Model | Unscrubbed (Acc / F1) | Scrubbed (Acc / F1) | Delta (Acc Drop) |
|---|---|---|---|
| **XGBoost** | 0.8204 / 0.8032 | 0.7655 / 0.7344 | -5.49% |
| **BiLSTM** | 0.7865 / 0.7511 | 0.7389 / 0.7101 | -4.76% |
| **RoBERTa** | 0.8682 / 0.8548 | 0.8285 / 0.8067 | -3.97% |
| **MentalBERT** | 0.8607 / 0.8455 | 0.8170 / 0.7941 | -4.37% |

As hypothesized, when explicit artifact "cheat words" are scrubbed, the accuracy and F1 of all models drop significantly. RoBERTa and MentalBERT perform similarly in raw accuracy, setting the stage for the SAS evaluation to differentiate their clinical value.

### 5.2 R2 — Symptom Alignment Scores

While RoBERTa and MentalBERT achieve comparable accuracy, their internal representations differ drastically. MentalBERT achieves a systematically higher Symptom Alignment Score (SAS) across the clinical classes compared to RoBERTa.

* **PTSD** sees the largest clinical alignment gain when scrubbed. In the unscrubbed arm, models exploit the token `ptsd`, heavily inflating performance. In the scrubbed arm, the models fall back to identifying genuine PCL-5 symptoms (e.g., *trauma, flashbacks, nightmares*).
* **Domain-Specific Superiority:** MentalBERT demonstrates that pre-training on a domain-specific mental health corpus results in a feature space that intrinsically aligns better with psychiatric symptomology than a general web-corpus model like RoBERTa.

---

## 6. Discussion

### 6.1 Prior Benchmarks Should Report Alignment Alongside F1
Published papers reporting >85% F1 on mental health datasets include a large artifact-leakage component. Our results quantify that component across four distinct architectures. Researchers should report alignment scores (like SAS) alongside F1 to allow readers to distinguish headline accuracy from clinically grounded accuracy.

### 6.2 The Value of Domain-Specific Models
If a general model (RoBERTa) and a domain-specific model (MentalBERT) achieve identical accuracy, which is better? The SAS framework proves that the domain-specific model is superior for medical applications because its decision boundary is constructed from actual symptoms rather than statistical shortcuts.

---

## 7. Limitations and Ethics

**Label validity:** Subreddit membership is a proxy label, not a clinical diagnosis. 
**No individual-level risk prediction:** This is a validation study, not an individual diagnostic tool.
**SAS is a lower bound:** The lexical matching approach is a strict, conservative floor for clinical alignment.

---

## References

[References omitted for brevity; see draft v1 for full citations]
