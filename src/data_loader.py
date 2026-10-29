"""Dataset loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO, BytesIO
from pathlib import Path

import pandas as pd


REQUIRED_TRAIN_COLUMNS = {"smiles", "target", "target_type"}
REQUIRED_TEST_COLUMNS = {"id", "smiles", "target_type"}
TARGET_TYPES = ["tg", "egc", "egb", "nc", "eps", "ei", "eea"]


@dataclass
class DatasetLoadResult:
    dataframe: pd.DataFrame
    source: str
    warnings: list[str]


def validate_columns(df: pd.DataFrame, required: set[str]) -> list[str]:
    return [column for column in required if column not in df.columns]


def load_csv(path: Path | None = None, uploaded_file=None) -> DatasetLoadResult:
    warnings: list[str] = []
    source = "upload"
    if uploaded_file is not None:
        raw = uploaded_file.read()
        uploaded_file.seek(0) if hasattr(uploaded_file, "seek") else None
        df = pd.read_csv(BytesIO(raw))
        source = getattr(uploaded_file, "name", "uploaded CSV")
    elif path is not None and Path(path).is_file():
        df = pd.read_csv(path)
        source = str(path)
    else:
        raise FileNotFoundError("No valid dataset source was provided.")

    if df.empty:
        warnings.append("The dataset contains no rows.")
    return DatasetLoadResult(df, source, warnings)


def summarize_dataset(df: pd.DataFrame, target_col: str = "target") -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "unique_smiles": int(df["smiles"].nunique(dropna=True)) if "smiles" in df.columns else 0,
        "target_counts": df["target_type"].value_counts(dropna=False).to_dict() if "target_type" in df.columns else {},
    }