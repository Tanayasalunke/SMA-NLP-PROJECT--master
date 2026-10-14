"""
Task 2: Build 4 clinical lexicons (PHQ-9, GAD-7, PCL-5, SPIN).
- Base terms derived ONLY from instrument item wording (not from SHAP output)
- POS-tagged so WordNet expansion stays within the correct word sense
- Expanded synonyms filtered to TF-IDF vocabulary (can't match a SHAP feature otherwise)
- Artifact list finalized
- Outputs: lexicons.pkl, lexicons_for_review.csv
"""
import os
import pickle
import csv
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

BASE      = os.path.dirname(os.path.abspath(__file__))
LEM       = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

WN_POS = {
    "n": wordnet.NOUN,
    "v": wordnet.VERB,
    "a": wordnet.ADJ,
}


# ── Load TF-IDF vocabularies ───────────────────────────────────────────────────
# Only terms present in at least one model vocabulary can ever appear as SHAP
# features, so restricting to this set is both practical and principled.
def load_vocab(fname):
    with open(os.path.join(BASE, fname), "rb") as f:
        saved = pickle.load(f)
    return set(saved["vectorizer"].get_feature_names_out())

print("Loading TF-IDF vocabularies...")
vocab = load_vocab("model_unscrubbed.pkl") | load_vocab("model_scrubbed.pkl")
print(f"  Combined vocabulary: {len(vocab)} terms\n")


# ── Clinical instrument base terms (word, POS) ─────────────────────────────────
# POS tags: n=noun, v=verb, a=adjective
# Derived from item wording of each instrument only.

PHQ9_BASE = [
    # Item 1: little interest or pleasure in doing things
    ("interest",      "n"), ("pleasure",   "n"), ("anhedonia",    "n"),
    # Item 2: feeling down, depressed, or hopeless
    ("depressed",     "a"), ("hopeless",   "a"), ("hopelessness", "n"),
    # Item 3: trouble sleeping / sleeping too much
    ("sleep",         "v"), ("insomnia",   "n"), ("sleepless",    "a"),
    # Item 4: tired or little energy
    ("tired",         "a"), ("fatigue",    "n"), ("exhausted",    "a"), ("energy", "n"),
    # Item 5: poor appetite or overeating
    ("appetite",      "n"), ("overeat",    "v"),
    # Item 6: feeling bad, failure, guilt
    ("failure",       "n"), ("worthless",  "a"), ("guilty",       "a"),
    ("guilt",         "n"), ("shame",      "n"),
    # Item 7: trouble concentrating
    ("concentrate",   "v"), ("concentration", "n"), ("focus",     "v"),
    # Item 8: slow / restless / fidgety / agitated
    ("restless",      "a"), ("restlessness","n"), ("agitated",    "a"),
    ("agitation",     "n"),
    # Item 9: thoughts of death / self-harm
    ("suicidal",      "a"), ("suicide",    "n"), ("death",        "n"),
    ("hurt",          "v"), ("harm",       "n"), ("kill",         "v"),
]

GAD7_BASE = [
    # Item 1: nervous, anxious, on edge
    ("nervous",       "a"), ("anxious",    "a"), ("nervousness",  "n"), ("tense",  "a"),
    # Item 2-3: worrying, can't stop
    ("worry",         "v"), ("worrying",   "n"), ("overthink",    "v"),
    # Item 4: trouble relaxing
    ("relax",         "v"), ("tension",    "n"),
    # Item 5: restless, hard to sit still
    ("restless",      "a"), ("restlessness","n"),
    # Item 6: irritable, easily annoyed
    ("irritable",     "a"), ("irritability","n"), ("annoyed",     "a"),
    # Item 7: afraid, something awful might happen
    ("afraid",        "a"), ("fear",       "n"), ("dread",        "v"),
    ("apprehensive",  "a"), ("panic",      "n"),
]

PCL5_BASE = [
    # Items 1-2: memories and dreams
    ("memory",        "n"), ("nightmare",  "n"), ("intrusion",    "n"), ("intrusive", "a"),
    # Item 3: flashbacks
    ("flashback",     "n"), ("relive",     "v"),
    # Items 4-5: distress when reminded
    ("trigger",       "v"), ("trauma",     "n"), ("traumatic",    "a"), ("distress", "n"),
    # Items 6-7: avoidance
    ("avoidance",     "n"), ("avoid",      "v"), ("numb",         "a"),
    # Item 8: trouble remembering
    ("dissociate",    "v"), ("dissociation","n"),
    # Items 9-10: negative beliefs, blame, guilt
    ("blame",         "v"), ("guilty",     "a"), ("shame",        "n"), ("guilt",    "n"),
    # Item 11: fear, horror, anger
    ("horror",        "n"), ("anger",      "n"), ("fear",         "n"),
    # Item 12: loss of interest
    ("detachment",    "n"),
    # Item 13: distant, cut off
    ("distant",       "a"), ("isolated",   "a"), ("isolation",    "n"), ("withdraw", "v"),
    # Item 14: inability to feel positive
    ("numb",          "a"), ("emptiness",  "n"),
    # Item 15: irritability, aggression
    ("irritable",     "a"), ("aggressive", "a"), ("aggression",   "n"), ("rage",     "n"),
    # Item 16: recklessness
    ("reckless",      "a"), ("recklessness","n"),
    # Item 17: hypervigilance
    ("hypervigilant", "a"), ("hypervigilance","n"), ("watchful",  "a"),
    # Item 18: startle response
    ("startle",       "v"), ("jumpy",      "a"),
    # Item 19: concentrating
    ("concentrate",   "v"), ("concentration","n"),
    # Item 20: sleep
    ("sleep",         "v"), ("insomnia",   "n"),
    # Core PTSD vocabulary
    ("abuse",         "n"), ("assault",    "n"), ("violence",     "n"),
    ("therapy",       "n"), ("therapist",  "n"), ("symptom",      "n"),
]

SPIN_BASE = [
    # Item 1: afraid of authority
    ("authority",     "n"),
    # Item 2: blushing
    ("blush",         "v"),
    # Items 3, 8, 9, 17: parties and social situations/events
    # "social" is explicitly core SPIN vocabulary (Social Phobia Inventory)
    # and the top SHAP feature for social_anxiety in both arms -- must be in lexicon
    ("social",        "a"), ("party",      "n"), ("event",        "n"),
    # Items 4, 10: talking to strangers
    ("stranger",      "n"), ("talk",       "v"), ("speak",        "v"),
    # Items 5, 12: being criticized
    ("criticism",     "n"), ("judge",      "v"), ("judgment",     "n"),
    # Item 6: embarrassment
    ("embarrass",     "v"), ("embarrassment","n"), ("humiliate",  "v"), ("humiliation","n"),
    # Item 7: sweating
    ("sweat",         "v"),
    # Item 9: center of attention
    ("attention",     "n"), ("spotlight",  "n"),
    # Item 11: public speaking
    ("speech",        "n"), ("performance","n"), ("presentation", "n"),
    # Items 13, 17: palpitations, trembling, sweating
    ("palpitation",   "n"), ("tremble",    "v"), ("shake",        "v"),
    # Item 14: people watching
    ("observe",       "v"),
    # Item 15: embarrassed, looking stupid
    ("foolish",       "a"), ("ridicule",   "v"),
    # General social anxiety vocabulary
    ("shy",           "a"), ("shyness",    "n"), ("awkward",      "a"),
    ("rejection",     "n"), ("judged",     "a"),
]


# ── Post-expansion exclusions (wrong WordNet sense, not in instrument wording) ─
# These appeared via WordNet expansion but are wrong-sense artifacts:
# e.g. "concentrate"->verb synset 1 gives "center/centre" (geometry sense, not focus)
#      "hurt"->verb gives "smart" and "ache" (physical pain, not PHQ-9 language)
#      "stranger"->noun gives "alien" (foreign national sense)
EXCLUDE_FROM = {
    "PHQ9": {"center", "centre", "smart", "ache", "injury"},
    "GAD7": {"unsatisfied"},
    "PCL5": {"divorce", "trip", "force", "madness"},
    "SPIN": {"address", "alien", "attending", "authorization",
             "discover", "find", "humble", "mind", "mouth", "unknown", "utter"},
}

# ── Artifact list ──────────────────────────────────────────────────────────────
ARTIFACT_LIST = [
    "depression", "anxiety", "ptsd", "social anxiety", "socialanxiety",
    "diagnosed", "diagnosis", "dae", "throwaway", "tw:", "trigger warning",
    "r/depression", "r/anxiety", "r/ptsd", "r/socialanxiety",
    "sa", "gad",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def lemmatize(word):
    w = LEM.lemmatize(word, pos="v")
    return LEM.lemmatize(w, pos="n")


def in_vocab(word):
    """True if the word or its lemma is in the combined TF-IDF vocabulary."""
    return word in vocab or lemmatize(word) in vocab


def expand_wordnet(tagged_terms, max_synonyms=4):
    """
    POS-constrained synonym expansion.
    - Only uses synsets matching the tagged POS (no wrong-sense synonyms)
    - Rejects multi-word expressions (underscores in WordNet)
    - Rejects stopwords and short tokens
    - Filters to TF-IDF vocabulary so only matchable terms are kept
    """
    base_words  = {t for t, _ in tagged_terms}
    expanded    = set(base_words)

    for term, pos in tagged_terms:
        wn_pos  = WN_POS[pos]
        synsets = wordnet.synsets(term, pos=wn_pos)[:1]   # first (most common) sense
        for syn in synsets:
            for lemma in syn.lemmas()[:max_synonyms]:
                raw = lemma.name()
                if "_" in raw:          # skip multi-word expressions
                    continue
                word = raw.lower()
                if (
                    len(word) > 3
                    and word not in STOPWORDS
                    and word.isalpha()
                    and in_vocab(word)  # must be learnable by the model
                ):
                    expanded.add(word)

    # Always keep base terms even if rare in training data
    for t, _ in tagged_terms:
        expanded.add(t.lower())

    return sorted(expanded)


def build_lexicon(tagged_terms, name, exclude=None):
    print(f"  {name}:")
    raw      = {t.lower() for t, _ in tagged_terms}
    expanded = expand_wordnet(tagged_terms)
    if exclude:
        expanded = [t for t in expanded if t not in exclude]
    in_v = sum(1 for t in expanded if in_vocab(t))
    print(f"    Base: {len(raw)}  |  After expansion + vocab filter + exclusions: {len(expanded)}  "
          f"({in_v} in vocab)")
    return sorted(expanded)


# ── Build ──────────────────────────────────────────────────────────────────────
print("Building lexicons...")
lexicons = {
    "PHQ9": build_lexicon(PHQ9_BASE, "PHQ-9 (depression)",    EXCLUDE_FROM["PHQ9"]),
    "GAD7": build_lexicon(GAD7_BASE, "GAD-7 (anxiety)",       EXCLUDE_FROM["GAD7"]),
    "PCL5": build_lexicon(PCL5_BASE, "PCL-5 (ptsd)",          EXCLUDE_FROM["PCL5"]),
    "SPIN": build_lexicon(SPIN_BASE, "SPIN (social_anxiety)",  EXCLUDE_FROM["SPIN"]),
}

print()
for name, terms in lexicons.items():
    print(f"{name} ({len(terms)} terms):")
    print("  " + ", ".join(terms))
    print()

print(f"Artifact list ({len(ARTIFACT_LIST)} terms):")
print("  " + ", ".join(ARTIFACT_LIST))


# ── Save ───────────────────────────────────────────────────────────────────────
out_pkl = os.path.join(BASE, "lexicons.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump({"lexicons": lexicons, "artifact_list": ARTIFACT_LIST}, f)
print(f"\nSaved {out_pkl}")

out_csv = os.path.join(BASE, "lexicons_for_review.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["instrument", "term", "rater1_valid", "rater2_valid", "notes"]
    )
    writer.writeheader()
    for instrument, terms in lexicons.items():
        for term in terms:
            writer.writerow({
                "instrument": instrument, "term": term,
                "rater1_valid": "", "rater2_valid": "", "notes": "",
            })

total = sum(len(t) for t in lexicons.values())
print(f"Saved {out_csv}  ({total} terms to validate)")
print("\nFill in lexicons_for_review.csv for inter-rater agreement, then run 05_alignment.py.")
