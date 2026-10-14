"""
Run the full pipeline end-to-end.
Steps 1-6 are fully automated. Step 7 requires human input first.

Usage:
  python run_all.py           -- runs steps 01-06
  python run_all.py --all     -- runs 01-07 (only after lexicons_for_review.csv is filled in)
"""
import subprocess, sys, os, time

BASE    = os.path.dirname(os.path.abspath(__file__))
PYTHON  = sys.executable
RUN_ALL = "--all" in sys.argv

STEPS = [
    ("01_pipeline.py",  "Merge -> dedup -> downsample -> split -> scrub"),
    ("02_train.py",     "Train TF-IDF + XGBoost (both arms) -- ~15 min"),
    ("02b_evaluate.py", "Save classification results + confusion matrices"),
    ("03_shap.py",      "SHAP explainability (both arms)    -- ~15 min"),
    ("04_lexicons.py",  "Build clinical lexicons"),
    ("05_alignment.py", "Alignment scoring + bootstrap CIs  -- ~2 min"),
    ("06_figures.py",   "Generate XGBoost paper figures"),
    ("08a_bilstm.py",      "Fine-tune BiLSTM (both arms)       -- ~30-60 min CPU"),
    ("08b_roberta.py",     "Fine-tune RoBERTa (both arms)      -- ~60-90 min CPU"),
    ("08c_mentalbert.py",  "Fine-tune MentalBERT (both arms)   -- ~60-90 min CPU"),
    ("09a_roberta_shap.py", "RoBERTa SHAP + alignment            -- ~15-30 min CPU"),
    ("09b_mentalbert_shap.py", "MentalBERT SHAP + alignment      -- ~15-30 min CPU"),
    ("10_figures_multimodel.py", "Model comparison figures"),
]
if RUN_ALL:
    STEPS.append(("07_interrater.py", "Inter-rater kappa + lexicon pruning"))

total = len(STEPS)
for i, (script, desc) in enumerate(STEPS, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{total}] {script}  --  {desc}")
    print('='*60)
    t0 = time.time()
    result = subprocess.run([PYTHON, os.path.join(BASE, script)], check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nFAILED at {script} (exit code {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    print(f"\n[{i}/{total}] Done in {elapsed:.0f}s")

print("\n" + "="*60)
print("Pipeline complete.")
if not RUN_ALL:
    print("\nNext steps:")
    print("  1. Fill in lexicons_for_review.csv (inter-rater review), then run:")
    print("     python run_all.py --all")
    print("  2. DistilBERT training (~2-4h CPU) saved to transformer_unscrubbed/ and transformer_scrubbed/")
    print("  3. Comparison figures saved to fig6_model_comparison.png")
