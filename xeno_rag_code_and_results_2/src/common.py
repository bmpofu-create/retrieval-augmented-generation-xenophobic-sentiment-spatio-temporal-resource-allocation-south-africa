"""Shared paths and loaders for the Xeno-RAG pipeline."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
DATA = ROOT / "SA_XenoDataset_model_ready.csv"
SEED = 42
CLASSES = ["Negative", "Neutral", "Positive"]

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["dt"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("dt", kind="stable").reset_index(drop=True)
    return df

def add_groups_and_folds(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Group identical texts together so duplicates never straddle train/test."""
    from sklearn.model_selection import StratifiedGroupKFold
    df = df.copy()
    df["grp"] = df["text_ml"].factorize()[0]
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(sgkf.split(df, df["Sentiment"], df["grp"])):
        df.loc[df.index[te], "fold"] = k
    return df
