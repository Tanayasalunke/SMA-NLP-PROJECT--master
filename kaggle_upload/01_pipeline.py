"""
Step 1-5: merge raw CSVs -> deduplicate -> downsample -> split -> scrub
Outputs: merged_raw.csv, merged_balanced.csv, train_df.csv, test_df.csv
"""
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
KEEP = ["subreddit", "author", "date", "post"]

FILES = {
    "depression_pre_features_tfidf_256.csv":    "depression",
    "anxiety_pre_features_tfidf_256.csv":       "anxiety",
    "ptsd_pre_features_tfidf_256.csv":          "ptsd",
    "socialanxiety_pre_features_tfidf_256.csv": "social_anxiety",
    "jokes_pre_features_tfidf_256.csv":         "control",
    "personalfinance_pre_features_tfidf_256.csv":"control",
    "teaching_pre_features_tfidf_256.csv":      "control",
}

ARTIFACT_TERMS = [
    "depression", "anxiety", "ptsd", "social anxiety", "socialanxiety",
    "diagnosed", "diagnosis", "dae", "throwaway", "tw:", "trigger warning",
    "r/depression", "r/anxiety", "r/ptsd", "r/socialanxiety",
]
ARTIFACT_TERMS_WB = ["sa", "gad"]


def scrub_text(text):
    text = str(text).lower()
    for term in ARTIFACT_TERMS:
        text = re.sub(re.escape(term), "", text)
    for term in ARTIFACT_TERMS_WB:
        text = re.sub(r"\b" + re.escape(term) + r"\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


# -- Step 1: Merge --------------------------------------------------------------
print("Step 1 - Merging CSVs...")
dfs = []
for fname, label in FILES.items():
    path = os.path.join(BASE, fname)
    d = pd.read_csv(path, usecols=lambda c: c in KEEP, engine="c", on_bad_lines="skip")
    assert set(KEEP).issubset(d.columns), f"Schema mismatch in {fname}"
    d["label"] = label
    dfs.append(d)
    print(f"  {fname}: {len(d)} rows")

data = pd.concat(dfs, ignore_index=True)
print(f"\nTotal after merge: {len(data)}")
print(data.label.value_counts().to_string())

data.to_csv(os.path.join(BASE, "merged_raw.csv"), index=False)
print("Saved merged_raw.csv\n")
assert len(data) == 102984, f"Expected 102984 rows, got {len(data)} - check file versions"

# -- Step 2: Deduplicate cross-class users --------------------------------------
print("Step 2 - Removing cross-class users...")
overlap = data.groupby("author")["label"].nunique()
leaky = overlap[overlap > 1].index
data = data[~data["author"].isin(leaky)].copy()
print(f"  Removed {len(leaky)} users -> {len(data)} rows")
print(data.label.value_counts().to_string())
assert len(data) == 96003, f"Expected 96003 rows after dedup, got {len(data)}"
print()

# -- Step 3: Downsample control -------------------------------------------------
print("Step 3 - Downsampling control class...")
CONTROL_TARGET = 25000
control_data = data[data.label == "control"]
non_control = data[data.label != "control"]

control_sampled = (
    control_data
    .groupby("subreddit", group_keys=False)[control_data.columns]
    .apply(
        lambda x: x.sample(
            n=min(len(x), CONTROL_TARGET // control_data.subreddit.nunique()),
            random_state=42
        ),
        include_groups=False
    )
)
if "subreddit" not in control_sampled.columns:
    control_sampled["subreddit"] = control_data.loc[control_sampled.index, "subreddit"]

data_balanced = pd.concat([non_control, control_sampled], ignore_index=True)
print(f"  {len(data_balanced)} rows after balancing")
print(data_balanced.label.value_counts().to_string())

data_balanced.to_csv(os.path.join(BASE, "merged_balanced.csv"), index=False)
print("Saved merged_balanced.csv\n")
assert len(data_balanced) == 70583, f"Expected 70583 rows, got {len(data_balanced)}"

# -- Step 4: Author-level train/test split --------------------------------------
print("Step 4 - Author-level train/test split (80/20)...")
authors = data_balanced["author"].unique()
train_authors, test_authors = train_test_split(authors, test_size=0.2, random_state=42)

train_df = data_balanced[data_balanced.author.isin(train_authors)].copy()
test_df  = data_balanced[data_balanced.author.isin(test_authors)].copy()

print(f"  Train: {len(train_df)}")
print(train_df.label.value_counts().to_string())
print(f"\n  Test: {len(test_df)}")
print(test_df.label.value_counts().to_string())
assert len(train_df) == 56465, f"Expected 56465 train rows, got {len(train_df)}"
assert len(test_df)  == 14118, f"Expected 14118 test rows, got {len(test_df)}"
print()

# -- Step 5: Scrubbed / Unscrubbed text arms ------------------------------------
print("Step 5 - Creating scrubbed / unscrubbed text arms...")
for df in [train_df, test_df]:
    df["text_unscrubbed"] = df["post"]
    df["text_scrubbed"]   = df["post"].apply(scrub_text)

train_df.to_csv(os.path.join(BASE, "train_df.csv"), index=False)
test_df.to_csv(os.path.join(BASE,  "test_df.csv"),  index=False)
print("Saved train_df.csv and test_df.csv")
print("\nPipeline complete. Run 02_train.py next.")
