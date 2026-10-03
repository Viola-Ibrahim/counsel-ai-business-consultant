"""
cleaning_validation.py
-----------------------
Responsible for cleaning the DataFrame based on issues found during profiling:
removing duplicates and handling missing values.
Returns the cleaned DataFrame plus a log of what was changed.

Design choice: a column is NEVER dropped just because it has missing values
(a null often means "not applicable", e.g. no return reason for an order that
was not returned). Numeric nulls stay as NaN so calculations ignore them
instead of using invented values.
"""

import numpy as np
import pandas as pd

MISSING_LABEL = "Missing"  # label used for nulls in text/categorical columns


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    log = {}

    # 1. Trim whitespace in text columns; empty strings become real nulls
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        df[col] = df[col].str.strip().replace("", np.nan)

    # 2. Remove fully duplicated rows
    rows_before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = rows_before - len(df)

    # 3. Drop only columns that are 100% empty (nothing to analyze)
    columns_to_drop = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=columns_to_drop)
    log["columns_dropped"] = columns_to_drop

    # 4. Handle remaining nulls WITHOUT deleting anything
    handled = {}
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        if null_pct == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            action = "kept as NaN"
        elif isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == "object" \
                or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("object").fillna(MISSING_LABEL)
            action = f"filled with '{MISSING_LABEL}'"
        else:
            action = "kept as NaN"

        handled[col] = {"null_pct": round(null_pct, 1), "action": action}

    log["null_handling"] = dict(sorted(handled.items(), key=lambda kv: -kv[1]["null_pct"]))

    return df, log