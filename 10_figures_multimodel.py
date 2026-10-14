import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":  150,
})

# Colors for models
C_XGB   = "#264653"  # Dark Teal
C_BILSTM = "#2a9d8f"  # Light Teal
C_ROBERTA = "#e9c46a" # Yellow
C_MENTAL = "#e76f51"  # Orange/Red

CLASS_ORDER = ["depression", "anxiety", "ptsd", "social_anxiety", "control"]
LABEL = {
    "anxiety":       "Anxiety",
    "control":       "Control",
    "depression":    "Depression",
    "ptsd":          "PTSD",
    "social_anxiety":"Social\nAnxiety",
}

def load(fname):
    try:
        with open(os.path.join(BASE, fname), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: {fname} not found!")
        return None

clf_xgb        = load("classification_results.pkl")
clf_bilstm     = load("classification_results_bilstm.pkl")
clf_roberta    = load("classification_results_roberta.pkl")
clf_mentalbert = load("classification_results_mentalbert.pkl")

align_xgb        = load("alignment_results.pkl")
align_roberta    = load("roberta_alignment.pkl")
align_mentalbert = load("mentalbert_alignment.pkl")

# We plot the unscrubbed and scrubbed macro-F1 overall, or F1 per class?
# The task says: "Panel A: Accuracy comparison (macro-F1) across all 4 models."
# It might be cleaner to plot Macro-F1 across all 4 models for both unscrubbed and scrubbed arms.
# And Panel B: SAS for XGBoost, RoBERTa, and MentalBERT.

# ── Figure 6 — two panels ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison: Performance and Clinical Alignment (k=50)",
             fontweight="bold", y=1.01)

# ── Panel (a): Macro-F1 summary ────────────────────────────────────────────────
ax = axes[0]
models = ["XGBoost", "BiLSTM", "RoBERTa", "MentalBERT"]
colors = [C_XGB, C_BILSTM, C_ROBERTA, C_MENTAL]
results = [clf_xgb, clf_bilstm, clf_roberta, clf_mentalbert]

x = np.arange(len(models))
w = 0.35

f1_u = [r["unscrubbed"]["report"]["macro avg"]["f1-score"] if r else 0 for r in results]
f1_s = [r["scrubbed"]["report"]["macro avg"]["f1-score"] if r else 0 for r in results]

ax.bar(x - w/2, f1_u, w, color="#cccccc", label="Unscrubbed", zorder=3)
ax.bar(x + w/2, f1_s, w, color=colors, label="Scrubbed", zorder=3)

# Add pattern to unscrubbed for distinction
for patch in ax.patches[:len(models)]:
    patch.set_hatch('//')
    patch.set_edgecolor('white')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Macro-F1 Score")
ax.set_title("(a) Classification Performance (Macro-F1)", fontsize=12, pad=8)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10, loc="lower right")

for i, (u, s) in enumerate(zip(f1_u, f1_s)):
    if u > 0:
        ax.text(x[i] - w/2, u + 0.01, f"{u:.3f}", ha="center", va="bottom", fontsize=9)
    if s > 0:
        ax.text(x[i] + w/2, s + 0.01, f"{s:.3f}", ha="center", va="bottom", fontsize=9)

# ── Panel (b): SAS comparison — scrubbed arm ─────────────────────────────────
ax = axes[1]
sas_models = ["XGBoost", "RoBERTa", "MentalBERT"]
sas_colors = [C_XGB, C_ROBERTA, C_MENTAL]
aligns     = [
    align_xgb["scores"]["scrubbed"] if align_xgb else None,
    align_roberta["scrubbed"] if align_roberta else None,
    align_mentalbert["scrubbed"] if align_mentalbert else None
]

x_sas = np.arange(len(CLASS_ORDER))
w_sas = 0.25

for i, (model_name, align, color) in enumerate(zip(sas_models, aligns, sas_colors)):
    sas_vals = [align[c]["SAS"] if align else 0 for c in CLASS_ORDER]
    pos = x_sas + (i - 1) * w_sas
    ax.bar(pos, sas_vals, w_sas, color=color, label=model_name, zorder=3)
    for p, v in zip(pos, sas_vals):
        if v > 0:
            ax.text(p, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

ax.set_xticks(x_sas)
ax.set_xticklabels([LABEL[c] for c in CLASS_ORDER], fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Symptom Alignment Score (SAS)")
ax.set_title("(b) Clinical Alignment (SAS) in Scrubbed Arm", fontsize=12, pad=8)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10, loc="upper right")

fig.tight_layout()
out = os.path.join(BASE, "fig6_model_comparison.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")


# ── Summary table to stdout ───────────────────────────────────────────────────
def macro(clf, arm):
    return clf[arm]["report"]["macro avg"]["f1-score"] if clf else 0.0

print("\n=== Macro-F1 Summary ===")
print(f"{'Model':<15} {'Unscrubbed':>12} {'Scrubbed':>10} {'Delta':>8}")
print("-" * 47)
for m_name, res in zip(models, results):
    if res:
        u = macro(res, 'unscrubbed')
        s = macro(res, 'scrubbed')
        d = s - u
        print(f"{m_name:<15} {u:>12.3f} {s:>10.3f} {d:>8.3f}")

print("\n=== SAS (scrubbed) Summary ===")
print(f"{'Class':<15} {'XGBoost':>10} {'RoBERTa':>10} {'MentalBERT':>12}")
print("-" * 50)
for c in CLASS_ORDER:
    xs = align_xgb["scores"]["scrubbed"][c]["SAS"] if align_xgb else 0.0
    rs = align_roberta["scrubbed"][c]["SAS"] if align_roberta else 0.0
    ms = align_mentalbert["scrubbed"][c]["SAS"] if align_mentalbert else 0.0
    print(f"{c:<15} {xs:>10.4f} {rs:>10.4f} {ms:>12.4f}")

print("\nDone.")
