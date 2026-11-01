"""Fine-tune DistilBERT on the same 5 folds (run on Colab/GPU; NOT executed in the build sandbox -
Hugging Face and PyPI were blocked there, so this file is untested).
pip install transformers datasets torch scikit-learn
python train_distilbert.py --csv SA_XenoDataset_model_ready.csv --out distilbert_oof.csv
"""
import argparse, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", default="distilbert_oof.csv")
ap.add_argument("--epochs", type=int, default=3); a = ap.parse_args()
df = pd.read_csv(a.csv); df["dt"] = pd.to_datetime(df["datetime"]); df = df.sort_values("dt", kind="stable").reset_index(drop=True)
L = {"Negative": 0, "Neutral": 1, "Positive": 2}; y = df.Sentiment.map(L).to_numpy(); grp = df.text_ml.factorize()[0]
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
class DS(torch.utils.data.Dataset):
    def __init__(s, t, y): s.e = tok(list(t), truncation=True, max_length=96, padding=True); s.y = y
    def __len__(s): return len(s.y)
    def __getitem__(s, i): return {**{k: torch.tensor(v[i]) for k, v in s.e.items()}, "labels": torch.tensor(int(s.y[i]))}
cnt = np.bincount(y); w = torch.tensor((len(y) / (3 * cnt)), dtype=torch.float)
class WT(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        lab = inputs.pop("labels"); out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(out.logits, lab, weight=w.to(out.logits.device)); return (loss, out) if return_outputs else loss
oof = np.zeros((len(df), 3)); sg = StratifiedGroupKFold(5, shuffle=True, random_state=42)  # same seed/split as common.add_groups_and_folds
for k, (tr, te) in enumerate(sg.split(df, df.Sentiment, grp)):
    m = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    args = TrainingArguments(f"db_fold{k}", num_train_epochs=a.epochs, per_device_train_batch_size=32, learning_rate=3e-5, weight_decay=0.01,
                             save_strategy="no", report_to=[], logging_steps=100)
    t = WT(model=m, args=args, train_dataset=DS(df.text_ml[tr], y[tr])); t.train()
    lg = t.predict(DS(df.text_ml[te], y[te])).predictions; oof[te] = torch.softmax(torch.tensor(lg), 1).numpy()
    print(k, "macroF1", f1_score(y[te], oof[te].argmax(1), average="macro"))
print("pooled macroF1", f1_score(y, oof.argmax(1), average="macro"), "acc", accuracy_score(y, oof.argmax(1)))
pd.DataFrame(oof, columns=["p_neg", "p_neu", "p_pos"]).assign(raw_id=df.raw_id).to_csv(a.out, index=False)
