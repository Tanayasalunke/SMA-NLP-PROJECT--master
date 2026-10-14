# Instructions for Writing the Final Paper

This document is a direct guide for whoever is assembling the final IEEE paper. Everything you need is already generated and sitting in this folder. 

Here is exactly what files to use, what images to insert, and where they belong in the paper structure.

---

## 1. The Core Text to Copy-Paste
Do not write the paper from scratch! Open the file named **`paper_draft_extended.md`** (located right here in the main folder). 

This file contains a massive, fully-written, journal-length IEEE paper. It includes the Abstract, Introduction, Literature Review, Methodology, Results, and formatted References. Use this as the foundation of your Word document or LaTeX file.

---

## 2. The Images (And Exactly Where to Put Them)
When you are building the paper in Microsoft Word, you need to insert the generated charts to prove the results. Here is exactly which images from this folder to use, and what section to put them in:

### Put this in the "Methodology" or "Dataset" Section:
*   **Image File:** `fig1_stacked.png`
*   **What it is:** This shows the distribution of our dataset before and after balancing the classes.
*   **Caption to write under it:** *Figure 1: Class distribution of the Reddit Mental Health Dataset (RMHD) before and after author-level deduplication.*

### Put this in the "Results - Data Leakage" Section:
*   **Image File:** `fig2_deltas.png`
*   **What it is:** This bar chart is critical. It shows that when we removed the explicit "cheat words" (Scrubbed vs Unscrubbed), the accuracy dropped for *all* models. This proves that current literature suffers from Data Leakage.
*   **Caption to write under it:** *Figure 2: Absolute degradation in Macro-F1 score across architectures when explicitly scrubbed of dataset artifacts, demonstrating baseline data leakage.*

### Put this in the "Results - Model Performance" Section:
*   **Image File:** `fig5_confusion.png`
*   **What it is:** These are the confusion matrices. It visually demonstrates exactly how many predictions each model got right vs wrong across the 5 mental health classes.
*   **Caption to write under it:** *Figure 3: Confusion matrices for the tested architectures demonstrating multi-class classification performance on the scrubbed dataset.*

### Put this in the "Explainable AI (SHAP)" Section:
*   **Image File:** `fig4_shap.png`
*   **What it is:** This shows the exact top words that the deep learning models used to make their decisions (extracted using game theory).
*   **Caption to write under it:** *Figure 4: SHAP feature importance extraction, highlighting the top predictive tokens utilized by the contextual transformer models.*

### Put this in the "Results - Symptom Alignment Score (SAS)" Section:
*   **Image File:** `fig6_model_comparison_REAL_FOR_SLIDES.png` (or `fig6_model_comparison.png`)
*   **What it is:** **This is the most important chart in the whole project.** It shows the novel metric we invented (the SAS). It visually proves that the domain-specific model (MentalBERT) aligns with clinical psychiatry significantly better than the general model (RoBERTa).
*   **Caption to write under it:** *Figure 5: Symptom Alignment Score (SAS). Clinical symptom alignment comparison demonstrating domain-specific MentalBERT significantly outperforming general contextual models.*

---

## 3. Other Helpful Documents
If your professor asks deep technical questions, or if you need to fill out assignment templates, refer to:
*   **`detailed implementation nlp.pdf`**: Tells you exactly what the CPU and GPU were doing under the hood.
*   **`SMA-NLP Project Outline.pdf`**: The high-level overview of the project structure.
