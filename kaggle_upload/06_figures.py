"""
Task 4: Generate the four core paper figures.
  Fig 1 -- Stacked bar: SAS/SPUR/AMB per class (both arms)
  Fig 2 -- Delta comparison: accuracy delta vs. alignment delta per class
  Fig 3 -- Cross-scale heatmap: M_{c,s} (both arms)
  Fig 4 -- Annotated SHAP bar: top features colour-coded clinical/spurious/ambiguous
Outputs: fig1_stacked.png, fig2_deltas.png, fig3_heatmap.png, fig4_shap.png
"""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

C_SAS  = "#2a9d8f"   # teal  -- clinical
C_SPUR = "#e76f51"   # coral -- spurious
C_AMB  = "#bdbdbd"   # gray  -- ambiguous
C_F1   = "#264653"   # dark  -- accuracy delta
C_SASD = "#e9c46a"   # gold  -- alignment delta

LABEL = {
    "anxiety":       "Anxiety",
    "control":       "Control",
    "depression":    "Depression",
    "ptsd":          "PTSD",
    "social_anxiety":"Social\nAnxiety",
}
INSTR_LABEL = {"PHQ9": "PHQ-9", "GAD7": "GAD-7", "PCL5": "PCL-5", "SPIN": "SPIN"}
CLASS_ORDER = ["depression", "anxiety", "ptsd", "social_anxiety", "control"]

# ── Load ───────────────────────────────────────────────────────────────────────
def load(fname):
    with open(os.path.join(BASE, fname), "rb") as f:
        return pickle.load(f)

align   = load("alignment_results.pkl")
scores  = align["scores"]
delta   = align["delta_sas"]
shap_d  = load("shap_data.pkl")
lex_d   = load("lexicons.pkl")
lexicons = lex_d["lexicons"]
artifacts = set(lex_d["artifact_list"])

# Load F1 scores from saved classification results
clf_res = load("classification_results.pkl")
F1_UNSCRUBBED = {c: clf_res["unscrubbed"]["report"][c]["f1-score"]
                 for c in CLASS_ORDER}
F1_SCRUBBED   = {c: clf_res["scrubbed"]["report"][c]["f1-score"]
                 for c in CLASS_ORDER}

from nltk.stem import WordNetLemmatizer
import nltk; nltk.download("wordnet", quiet=True)
LEM = WordNetLemmatizer()
def lemmatize(w):
    w = LEM.lemmatize(w.lower(), "v")
    return LEM.lemmatize(w, "n")
def make_set(terms):
    s = set()
    for t in terms:
        s.add(t.lower()); s.add(lemmatize(t))
    return s
lex_sets = {k: make_set(v) for k, v in lexicons.items()}
art_set  = make_set(list(artifacts))
CLASS_TO_INSTR = {"depression":"PHQ9","anxiety":"GAD7","ptsd":"PCL5","social_anxiety":"SPIN","control":None}

def feature_color(feat, cls):
    L = lex_sets.get(CLASS_TO_INSTR.get(cls,""), set())
    f = feat.lower()
    in_L = f in L or lemmatize(f) in L
    in_A = f in art_set or lemmatize(f) in art_set
    if in_L:   return C_SAS,  "Clinical"
    if in_A:   return C_SPUR, "Spurious"
    return C_AMB, "Ambiguous"


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 -- Stacked bar: SAS / SPUR / AMB
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
fig.suptitle("Symptom Alignment Decomposition (k=50)", fontweight="bold", y=1.01)

for ax, arm, title in zip(axes, ("unscrubbed","scrubbed"), ("(a) Unscrubbed","(b) Scrubbed")):
    x = np.arange(len(CLASS_ORDER))
    sas  = [scores[arm][c]["SAS"]  for c in CLASS_ORDER]
    spur = [scores[arm][c]["SPUR"] for c in CLASS_ORDER]
    amb  = [scores[arm][c]["AMB"]  for c in CLASS_ORDER]

    ax.bar(x, sas,              color=C_SAS,  label="Clinical (SAS)",   zorder=3)
    ax.bar(x, spur, bottom=sas, color=C_SPUR, label="Spurious (SPUR)",  zorder=3)
    ax.bar(x, amb,  bottom=[s+p for s,p in zip(sas,spur)],
           color=C_AMB, label="Ambiguous (AMB)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in CLASS_ORDER], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion of SHAP mass" if arm=="unscrubbed" else "")
    ax.set_title(title, fontsize=12, pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # value labels on SAS bars
    for xi, v in zip(x, sas):
        if v > 0.02:
            ax.text(xi, v/2, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

handles = [mpatches.Patch(color=c, label=l)
           for c, l in [(C_SAS,"Clinical (SAS)"),(C_SPUR,"Spurious (SPUR)"),(C_AMB,"Ambiguous (AMB)")]]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5,-0.08), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(BASE,"fig1_stacked.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig1_stacked.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 -- Delta: F1 drop vs. SAS gain after scrubbing
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.5))
x   = np.arange(len(CLASS_ORDER))
w   = 0.35

f1_delta  = [F1_SCRUBBED[c] - F1_UNSCRUBBED[c] for c in CLASS_ORDER]
sas_delta = [delta[c] for c in CLASS_ORDER]

b1 = ax.bar(x - w/2, f1_delta,  width=w, color=C_F1,  label="F1 delta (scrubbed - unscrubbed)", zorder=3)
b2 = ax.bar(x + w/2, sas_delta, width=w, color=C_SASD, label="SAS delta (scrubbed - unscrubbed)", zorder=3)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([LABEL[c] for c in CLASS_ORDER], fontsize=10)
ax.set_ylabel("Delta (scrubbed - unscrubbed)")
ax.set_title("After Scrubbing: Accuracy Falls, Clinical Alignment Rises",
             fontweight="bold", pad=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10)

for bar in b1:
    v = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, v-0.005, f"{v:.3f}",
            ha="center", va="top", fontsize=8, color="white", fontweight="bold")
for bar in b2:
    v = bar.get_height()
    if abs(v) > 0.002:
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"+{v:.3f}",
                ha="center", va="bottom", fontsize=8, color="#7a6000", fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(BASE,"fig2_deltas.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig2_deltas.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 -- Cross-scale alignment heatmap M_{c,s}
# ══════════════════════════════════════════════════════════════════════════════
instr_order = ["PHQ9","GAD7","PCL5","SPIN"]
instr_labels = [INSTR_LABEL[i] for i in instr_order]
class_labels = [LABEL[c].replace("\n"," ") for c in CLASS_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
fig.suptitle("Cross-Scale Alignment Matrix M$_{c,s}$ (k=50)", fontweight="bold", y=1.02)

for ax, arm, title in zip(axes, ("unscrubbed","scrubbed"), ("(a) Unscrubbed","(b) Scrubbed")):
    mat = np.array([[scores[arm][c]["cross"][s] for s in instr_order] for c in CLASS_ORDER])
    sns.heatmap(mat, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=instr_labels, yticklabels=class_labels if arm=="unscrubbed" else [],
                vmin=0, vmax=mat.max()*1.1, linewidths=0.5, linecolor="#dddddd",
                cbar=arm=="scrubbed", cbar_kws={"shrink":0.8})
    ax.set_title(title, fontsize=12, pad=8)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Bold diagonal entries
    class_to_instr = {"depression":"PHQ9","anxiety":"GAD7","ptsd":"PCL5","social_anxiety":"SPIN","control":None}
    for ri, cls in enumerate(CLASS_ORDER):
        ci_inst = class_to_instr.get(cls)
        if ci_inst and ci_inst in instr_order:
            ci = instr_order.index(ci_inst)
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=False, edgecolor="#2a9d8f", lw=2.5))

fig.tight_layout()
fig.savefig(os.path.join(BASE,"fig3_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig3_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 -- Annotated SHAP bars: top-20 features, colour = clinical/spurious/ambiguous
# Show PTSD (most dramatic story) and Anxiety (GAD-7 alignment)
# ══════════════════════════════════════════════════════════════════════════════
SHOW_CLASSES = ["ptsd", "anxiety"]
SHOW_K = 20

fig, axes = plt.subplots(len(SHOW_CLASSES), 2,
                         figsize=(13, 4.5 * len(SHOW_CLASSES)))
fig.suptitle("Top-20 SHAP Features: Clinical / Spurious / Ambiguous",
             fontweight="bold", y=1.01)

for row, cls in enumerate(SHOW_CLASSES):
    for col, arm in enumerate(("unscrubbed", "scrubbed")):
        ax = axes[row][col]
        feat_names = shap_d[arm][cls]["top_feature_names"][:SHOW_K]
        mean_vals  = shap_d[arm][cls]["top_mean_abs_shap"][:SHOW_K]

        colors = [feature_color(f, cls)[0] for f in feat_names]
        y = np.arange(SHOW_K)[::-1]

        ax.barh(y, mean_vals, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(feat_names, fontsize=9)
        ax.set_xlabel("Mean |SHAP|", fontsize=9)
        ax.set_title(f"{LABEL[cls].replace(chr(10),' ')} -- {arm}", fontsize=11, pad=6)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

legend_handles = [
    mpatches.Patch(color=C_SAS,  label="Clinical (in instrument lexicon)"),
    mpatches.Patch(color=C_SPUR, label="Spurious (in artifact list)"),
    mpatches.Patch(color=C_AMB,  label="Ambiguous"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5,-0.04), frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(BASE,"fig4_shap.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig4_shap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 -- Confusion matrices (unscrubbed and scrubbed side by side)
# ══════════════════════════════════════════════════════════════════════════════
classes    = clf_res["unscrubbed"]["classes"]
cls_labels = [LABEL[c].replace("\n", " ") for c in classes]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Confusion Matrices (rows=true, columns=predicted)",
             fontweight="bold", y=1.02)

for ax, arm, title in zip(axes,
                           ("unscrubbed", "scrubbed"),
                           ("(a) Unscrubbed", "(b) Scrubbed")):
    cm = clf_res[arm]["cm"].astype(float)
    # Row-normalise so colour shows recall not raw counts
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, ax=ax, annot=cm.astype(int), fmt="d",
                cmap="Blues", vmin=0, vmax=1,
                xticklabels=cls_labels, yticklabels=cls_labels if arm=="unscrubbed" else [],
                linewidths=0.4, linecolor="#dddddd",
                cbar=arm=="scrubbed", cbar_kws={"shrink": 0.8, "label": "Row-normalised recall"})
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Predicted", fontsize=10)
    if arm == "unscrubbed":
        ax.set_ylabel("True", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

fig.tight_layout()
fig.savefig(os.path.join(BASE, "fig5_confusion.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig5_confusion.png")

print("\nAll 5 figures saved. Done.")
