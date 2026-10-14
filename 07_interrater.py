"""
Inter-rater agreement + lexicon pruning.
Run AFTER both raters have filled in lexicons_for_review.csv (columns rater1_valid,
rater2_valid: 1 = keep, 0 = remove).

Steps:
  1. Compute percent agreement and Cohen's kappa per instrument and overall
  2. Keep only terms where BOTH raters marked 1
  3. Overwrite lexicons.pkl with the pruned version
  4. Print stats and remind user to re-run 05_alignment.py + 06_figures.py
"""
import os, pickle, csv
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


# ── Load ratings ───────────────────────────────────────────────────────────────
csv_path = os.path.join(BASE, "lexicons_for_review.csv")
rows = []
with open(csv_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        r1 = row["rater1_valid"].strip()
        r2 = row["rater2_valid"].strip()
        if r1 == "" or r2 == "":
            print(f"WARNING: '{row['term']}' ({row['instrument']}) has a blank rating — skipping")
            continue
        rows.append({
            "instrument": row["instrument"],
            "term":       row["term"],
            "r1":         int(r1),
            "r2":         int(r2),
        })

if not rows:
    print("No completed ratings found. Fill in both rater columns first.")
    raise SystemExit(1)

print(f"Loaded {len(rows)} rated terms.\n")


# ── Cohen's kappa ──────────────────────────────────────────────────────────────
def cohens_kappa(r1_list, r2_list):
    r1 = np.array(r1_list)
    r2 = np.array(r2_list)
    n  = len(r1)
    po = (r1 == r2).mean()                      # observed agreement
    p1 = ((r1.sum() + r2.sum()) / (2 * n))      # marginal prob of 1
    p0 = 1 - p1                                  # marginal prob of 0
    pe = p1**2 + p0**2                           # expected agreement by chance
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return float(po), float(kappa)


print("=" * 55)
print(f"{'Instrument':12s}  {'N':>4}  {'%Agree':>7}  {'Kappa':>7}")
print("=" * 55)

instruments = sorted(set(r["instrument"] for r in rows))
keep_terms  = {inst: [] for inst in instruments}

for inst in instruments:
    inst_rows = [r for r in rows if r["instrument"] == inst]
    r1 = [r["r1"] for r in inst_rows]
    r2 = [r["r2"] for r in inst_rows]
    po, kappa = cohens_kappa(r1, r2)
    print(f"  {inst:10s}  {len(inst_rows):4d}  {po*100:6.1f}%  {kappa:7.3f}")
    keep_terms[inst] = [r["term"] for r in inst_rows if r["r1"] == 1 and r["r2"] == 1]

all_r1 = [r["r1"] for r in rows]
all_r2 = [r["r2"] for r in rows]
po_all, kappa_all = cohens_kappa(all_r1, all_r2)
print("-" * 55)
print(f"  {'OVERALL':10s}  {len(rows):4d}  {po_all*100:6.1f}%  {kappa_all:7.3f}")
print("=" * 55)

# Kappa interpretation
if   kappa_all >= 0.80: interp = "almost perfect"
elif kappa_all >= 0.61: interp = "substantial"
elif kappa_all >= 0.41: interp = "moderate"
elif kappa_all >= 0.21: interp = "fair"
else:                    interp = "slight"
print(f"\nOverall kappa = {kappa_all:.3f} ({interp} agreement)")

print("\nPruned lexicon sizes (terms kept by both raters):")
for inst, terms in keep_terms.items():
    print(f"  {inst}: {len(terms)} terms")


# ── Overwrite lexicons.pkl with pruned lexicons ────────────────────────────────
lex_path = os.path.join(BASE, "lexicons.pkl")
with open(lex_path, "rb") as f:
    lex_data = pickle.load(f)

lex_data["lexicons"] = keep_terms
lex_data["kappa"]    = kappa_all
lex_data["pct_agree"] = po_all

with open(lex_path, "wb") as f:
    pickle.dump(lex_data, f)

print(f"\nUpdated {lex_path} with pruned lexicons.")
print("\nNext steps:")
print("  python 05_alignment.py   (re-run with pruned lexicons)")
print("  python 06_figures.py     (regenerate figures with final numbers)")
