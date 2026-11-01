"""Binary hostile labels for the zone task.

Hostile is the existing Negative class. Neutral and Positive stay non-hostile.
A text that carries both labels is a conflict: those rows are left out of
training and out of the zone panel. This module does not rewrite the source CSV.
"""
import pandas as pd


def with_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out["text_ml"].fillna("").astype(str)
    out["text_ml"] = text
    out["hostile"] = (out["Sentiment"] == "Negative").astype(int)
    out["binary_conflict"] = out.groupby("text_ml")["hostile"].transform("nunique") > 1
    out["three_class_conflict"] = out.groupby("text_ml")["Sentiment"].transform("nunique") > 1
    return out
