"""
Task 3: Alignment scoring -- SAS, SPUR, AMB, cross-scale matrix, bootstrap CIs.

Formulas (weighted, SHAP-mass-normalised):
  SAS_c   = sum_{f in F_c^k} phi(f)*m(f,L_c)     / sum phi(f)
  SPUR_c  = sum_{f in F_c^k} phi(f)*m(f,A)*(1-m(f,L_c)) / sum phi(f)
  AMB_c   = 1 - SAS_c - SPUR_c
  M_{c,s} = sum_{f in F_c^k} phi(f)*m(f,L_s)     / sum phi(f)
  dSAS_c  = SAS_c(scrubbed) - SAS_c(unscrubbed)

Outputs: alignment_results.pkl, alignment_report.txt
"""
import os
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download("wordnet", quiet=True)

BASE  = os.path.dirname(os.path.abspath(__file__))
LEM   = WordNetLemmatizer()
K_PRIMARY   = 50
K_SENSITIVITY = [20, 50, 100]
N_BOOTSTRAP   = 1000
SEED          = 42
rng           = np.random.default_rng(SEED)

CLASS_TO_INSTRUMENT = {
    "depression":    "PHQ9",
    "anxiety":       "GAD7",
    "ptsd":          "PCL5",
    "social_anxiety":"SPIN",
    "control":       None,
}


# ── Load data ──────────────────────────────────────────────────────────────────
def load(fname):
    with open(os.path.join(BASE, fname), "rb") as f:
        return pickle.load(f)

print("Loading shap_data.pkl and lexicons.pkl...")
shap_data = load("shap_data.pkl")
lex_data  = load("lexicons.pkl")
lexicons  = lex_data["lexicons"]          # {"PHQ9": [...], "GAD7": [...], ...}
artifacts = set(lex_data["artifact_list"])


# ── Lemma-based matching ───────────────────────────────────────────────────────
def lemmatize(word):
    w = LEM.lemmatize(word.lower(), "v")
    return LEM.lemmatize(w, "n")

def match(feature, term_set):
    """1 if feature (or its lemma) is in term_set."""
    f = feature.lower()
    return int(f in term_set or lemmatize(f) in term_set)

def make_lexicon_set(terms):
    """Return a set containing both original terms and their lemmas."""
    s = set()
    for t in terms:
        s.add(t.lower())
        s.add(lemmatize(t))
    return s

# Pre-build lemmatized lexicon/artifact sets
lex_sets  = {k: make_lexicon_set(v) for k, v in lexicons.items()}
art_set   = make_lexicon_set(list(artifacts))

# Compute lexicon overlap (|L_c n L_s|) -- needed for paper caveat
print("\nLexicon pairwise overlap:")
instr = list(lex_sets.keys())
for i, a in enumerate(instr):
    for b in instr[i+1:]:
        overlap = lex_sets[a] & lex_sets[b]
        if overlap:
            print(f"  {a} n {b} = {len(overlap)} terms: {sorted(overlap)[:10]}{'...' if len(overlap)>10 else ''}")


# ── Core scoring function ──────────────────────────────────────────────────────
def score_one(class_name, feature_names, mean_shap, k):
    """
    Compute SAS, SPUR, AMB, and cross-scale row for one class at cutoff k.
    feature_names: array of feature names (top-200 ordered by mean |SHAP|)
    mean_shap:     array of mean |SHAP| values (same order)
    """
    feat_k  = feature_names[:k]
    phi_k   = mean_shap[:k]
    total   = phi_k.sum()
    if total == 0:
        return None

    inst = CLASS_TO_INSTRUMENT.get(class_name)
    L    = lex_sets.get(inst, set()) if inst else set()
    A    = art_set

    m_L  = np.array([match(f, L) for f in feat_k])
    m_A  = np.array([match(f, A) for f in feat_k])

    sas  = float((phi_k * m_L).sum() / total)
    spur = float((phi_k * m_A * (1 - m_L)).sum() / total)
    amb  = 1.0 - sas - spur

    # Unweighted variant (appendix): simple proportion of top-k features matching
    sas_uw  = float(m_L.sum() / k)
    spur_uw = float((m_A * (1 - m_L)).sum() / k)
    amb_uw  = 1.0 - sas_uw - spur_uw

    # Cross-scale row M_{c,s}
    cross = {s: float((phi_k * np.array([match(f, lex_sets[s]) for f in feat_k])).sum() / total)
             for s in lex_sets}

    return {"SAS": sas, "SPUR": spur, "AMB": amb,
            "SAS_uw": sas_uw, "SPUR_uw": spur_uw, "AMB_uw": amb_uw,
            "cross": cross}


# ── Bootstrap ─────────────────────────────────────────────────────────────────
def bootstrap_scores(class_name, feature_names, sample_shap, k, n=N_BOOTSTRAP):
    """
    Resample the 2000-row SHAP sample with replacement, recompute mean |SHAP|,
    then recompute SAS/SPUR/AMB at cutoff k.
    Returns arrays of shape (n,) for SAS, SPUR, AMB.
    """
    n_samples = sample_shap.shape[0]
    sas_boot, spur_boot, amb_boot = [], [], []

    inst = CLASS_TO_INSTRUMENT.get(class_name)
    L    = lex_sets.get(inst, set()) if inst else set()
    A    = art_set

    # Precompute match vectors (fixed for given feature set)
    m_L = np.array([match(f, L) for f in feature_names[:k]])
    m_A = np.array([match(f, A) for f in feature_names[:k]])

    for _ in range(n):
        idx      = rng.integers(0, n_samples, size=n_samples)
        phi_boot = sample_shap[idx, :k].mean(axis=0).astype(np.float64)
        total    = phi_boot.sum()
        if total == 0:
            continue
        sas  = (phi_boot * m_L).sum() / total
        spur = (phi_boot * m_A * (1 - m_L)).sum() / total
        amb  = 1.0 - sas - spur
        sas_boot.append(sas); spur_boot.append(spur); amb_boot.append(amb)

    def ci(arr):
        a = np.array(arr)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return {"SAS_ci": ci(sas_boot), "SPUR_ci": ci(spur_boot), "AMB_ci": ci(amb_boot)}


# ── Run scoring for both arms ──────────────────────────────────────────────────
results = {}

for arm in ("unscrubbed", "scrubbed"):
    print(f"\n{'='*60}")
    print(f"ARM: {arm}  (k={K_PRIMARY})")
    print('='*60)
    arm_data    = shap_data[arm]
    classes     = arm_data["classes"]
    arm_results = {}

    for cls in classes:
        feat_names  = arm_data[cls]["top_feature_names"]
        mean_abs    = arm_data[cls]["top_mean_abs_shap"]
        sample_abs  = arm_data[cls]["top_sample_shap"]

        scores = score_one(cls, feat_names, mean_abs, k=K_PRIMARY)
        cis    = bootstrap_scores(cls, feat_names, sample_abs, k=K_PRIMARY)

        # Sensitivity at k ∈ {20, 50, 100}
        sensitivity = {k: score_one(cls, feat_names, mean_abs, k=k)
                       for k in K_SENSITIVITY}

        arm_results[cls] = {**scores, **cis, "sensitivity": sensitivity,
                            "top_features": list(zip(feat_names[:K_PRIMARY],
                                                     mean_abs[:K_PRIMARY].tolist()))}

        print(f"  {cls:15s}  SAS={scores['SAS']:.3f}  SPUR={scores['SPUR']:.3f}  "
              f"AMB={scores['AMB']:.3f}  "
              f"95%CI SAS=[{cis['SAS_ci'][0]:.3f},{cis['SAS_ci'][1]:.3f}]")

    results[arm] = arm_results


# ── Delta SAS (scrubbed - unscrubbed) ─────────────────────────────────────────
print(f"\n{'='*60}")
print("Delta SAS (scrubbed - unscrubbed)")
print('='*60)
classes = shap_data["unscrubbed"]["classes"]
delta   = {}
for cls in classes:
    d = results["scrubbed"][cls]["SAS"] - results["unscrubbed"][cls]["SAS"]
    delta[cls] = d
    print(f"  {cls:15s}  dSAS={d:+.3f}")


# ── Cross-scale confusion matrix ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Cross-scale matrix M_{{c,s}} -- unscrubbed, k={K_PRIMARY}")
print('='*60)
instruments = list(lex_sets.keys())
header = f"{'class':15s}" + "".join(f"  {s:6s}" for s in instruments)
print(header)
for cls in classes:
    row = results["unscrubbed"][cls]["cross"]
    vals = "".join(f"  {row[s]:.3f}" for s in instruments)
    print(f"  {cls:15s}{vals}")

print(f"\nCross-scale matrix -- scrubbed, k={K_PRIMARY}")
print(header)
for cls in classes:
    row = results["scrubbed"][cls]["cross"]
    vals = "".join(f"  {row[s]:.3f}" for s in instruments)
    print(f"  {cls:15s}{vals}")


# ── Save results ───────────────────────────────────────────────────────────────
out_pkl = os.path.join(BASE, "alignment_results.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump({"scores": results, "delta_sas": delta}, f)
print(f"\nSaved {out_pkl}")


# ── Write plain-text report ────────────────────────────────────────────────────
report_lines = []
def rpt(line=""):
    report_lines.append(line)
    print(line)

rpt("\n" + "="*70)
rpt("ALIGNMENT REPORT")
rpt("="*70)

for arm in ("unscrubbed", "scrubbed"):
    rpt(f"\n-- {arm.upper()} ARM  (primary k={K_PRIMARY}, weighted) --")
    rpt(f"{'Class':15s}  {'SAS':>6}  {'SPUR':>6}  {'AMB':>6}  "
        f"{'SAS 95%CI':>18}  {'SPUR 95%CI':>18}")
    for cls in classes:
        r = results[arm][cls]
        rpt(f"  {cls:13s}  {r['SAS']:6.3f}  {r['SPUR']:6.3f}  {r['AMB']:6.3f}  "
            f"  [{r['SAS_ci'][0]:.3f},{r['SAS_ci'][1]:.3f}]"
            f"  [{r['SPUR_ci'][0]:.3f},{r['SPUR_ci'][1]:.3f}]")

rpt(f"\n-- UNWEIGHTED VARIANT (appendix, k={K_PRIMARY}) --")
rpt(f"{'Class':15s}  {'SAS_uw':>8}  {'SPUR_uw':>8}  {'AMB_uw':>8}  (unscrubbed / scrubbed)")
for cls in classes:
    u = results["unscrubbed"][cls]
    s = results["scrubbed"][cls]
    rpt(f"  {cls:13s}  {u['SAS_uw']:6.3f} / {s['SAS_uw']:6.3f}"
        f"   {u['SPUR_uw']:6.3f} / {s['SPUR_uw']:6.3f}"
        f"   {u['AMB_uw']:6.3f} / {s['AMB_uw']:6.3f}")

rpt(f"\n-- DELTA SAS (scrubbed - unscrubbed) --")
for cls in classes:
    rpt(f"  {cls:15s}  {delta[cls]:+.3f}")

rpt(f"\n-- SENSITIVITY (SAS at k=20,50,100 -- unscrubbed) --")
rpt(f"{'Class':15s}  {'k=20':>6}  {'k=50':>6}  {'k=100':>6}")
for cls in classes:
    s = results["unscrubbed"][cls]["sensitivity"]
    vals = "  ".join(f"{s[k]['SAS']:6.3f}" for k in [20, 50, 100])
    rpt(f"  {cls:15s}  {vals}")

rpt(f"\n-- SENSITIVITY (SAS at k=20,50,100 -- scrubbed) --")
rpt(f"{'Class':15s}  {'k=20':>6}  {'k=50':>6}  {'k=100':>6}")
for cls in classes:
    s = results["scrubbed"][cls]["sensitivity"]
    vals = "  ".join(f"{s[k]['SAS']:6.3f}" for k in [20, 50, 100])
    rpt(f"  {cls:15s}  {vals}")

out_txt = os.path.join(BASE, "alignment_report.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\nSaved {out_txt}")
print("Alignment scoring complete. Run 06_figures.py next.")
