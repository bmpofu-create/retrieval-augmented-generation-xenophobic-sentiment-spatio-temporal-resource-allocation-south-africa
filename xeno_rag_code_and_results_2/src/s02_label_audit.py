"""Audit sentiment labels and write a 200-tweet review sheet.

Does not relabel anything. Conflicts are reported so the binary model can
exclude them. The review sheet is for a person to fill in `review_label`.
"""
import json
import numpy as np
import pandas as pd
from common import *
from labels import with_binary_labels
from lexicon import category_frame

df = with_binary_labels(load_raw())
empty = df["text_ml"].str.strip().eq("")
lex = category_frame(df["text_ml"].tolist())
df = pd.concat([df.reset_index(drop=True), lex], axis=1)
hostile_lex = df[["derogatory", "threat", "expulsion", "mobilisation"]].sum(axis=1)

n_conflict_texts = int(df.loc[df.binary_conflict, "text_ml"].nunique())
summary = {
    "rows": int(len(df)),
    "empty_text": int(empty.sum()),
    "hostile": int(df.hostile.sum()),
    "non_hostile": int((df.hostile == 0).sum()),
    "sentiment_counts": df.Sentiment.value_counts().to_dict(),
    "binary_conflict_rows": int(df.binary_conflict.sum()),
    "binary_conflict_texts": n_conflict_texts,
    "three_class_conflict_rows": int(df.three_class_conflict.sum()),
    "three_class_conflict_texts": int(df.loc[df.three_class_conflict, "text_ml"].nunique()),
    "nonhostile_with_hostile_lexicon": int(((df.hostile == 0) & (hostile_lex > 0)).sum()),
    "hostile_without_hostile_lexicon": int(((df.hostile == 1) & (hostile_lex == 0)).sum()),
    "labels_rewritten": False,
    "note": "No tweet was relabelled. Binary training drops binary-conflict texts. Fill review_label in label_review_sample.csv before treating any new label as ground truth.",
}
json.dump(summary, open(OUT / "label_audit_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)

conflicts = df[df.binary_conflict].sort_values(["text_ml", "raw_id"])
conflicts[["raw_id", "datetime", "Tweet Origin", "Sentiment", "CleanedTweet"]].to_csv(
    OUT / "label_conflicts.csv", index=False)

rng = np.random.default_rng(SEED)
parts = []
conf_ids = df.index[df.binary_conflict]
parts.append(df.loc[conf_ids].assign(reason="binary_conflict"))

remain = 200 - len(parts[0])
hard_pos = df[(df.hostile == 0) & (hostile_lex > 0) & ~df.binary_conflict].sort_values(
    ["severity_raw", "raw_id"], ascending=[False, True])
take_pos = hard_pos.head(min(120, max(remain, 0)))
parts.append(take_pos.assign(reason="nonhostile_with_hostile_lexicon"))
remain = 200 - sum(len(p) for p in parts)

hard_neg = df[(df.hostile == 1) & (hostile_lex == 0) & ~df.binary_conflict].sort_values("raw_id")
if len(hard_neg) > 60:
    hard_neg = hard_neg.sample(60, random_state=SEED)
parts.append(hard_neg.head(max(remain, 0)).assign(reason="hostile_without_hostile_lexicon"))
remain = 200 - sum(len(p) for p in parts)

used = pd.concat(parts).index
filler_pool = df[(df.hostile == 0) & ~df.index.isin(used)]
if remain > 0 and len(filler_pool):
    filler = filler_pool.sample(min(remain, len(filler_pool)), random_state=SEED)
    parts.append(filler.assign(reason="other_nonhostile"))

sample = pd.concat(parts, ignore_index=True).head(200)
sample["review_label"] = ""
sample["notes"] = ""
cols = ["raw_id", "datetime", "Tweet Origin", "Province", "Sentiment", "reason",
        "severity_raw", "CleanedTweet", "review_label", "notes"]
sample[cols].to_csv(OUT / "label_review_sample.csv", index=False)
print("review rows", len(sample), "by reason", sample.reason.value_counts().to_dict(), flush=True)
