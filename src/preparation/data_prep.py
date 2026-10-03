"""
data_prep.py
-------------
LLM-driven data preparation layer.

The LLM looks at compact column statistics (never the full data) and DECIDES:
  - the role of every column (identifier, date, metric, dimension, free text, sensitive, other)
  - how to handle nulls in every column (based on the meaning of the column)
  - which columns are related to each other

Plain pandas then EXECUTES the plan. Every LLM decision passes through code guards,
so a bad decision can never delete columns or invent large amounts of data.
"""

import json
import re
import time
import warnings
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from groq import BadRequestError, RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

Role = Literal["identifier", "date", "metric", "dimension", "free_text", "sensitive", "other"]
NullStrategy = Literal["keep_nan", "fill_median", "fill_zero", "fill_label", "drop_rows"]

MAX_MEDIAN_FILL_PCT = 20.0   # never invent values for columns with more nulls than this
MAX_DROP_ROWS_PCT = 1.0      # never delete rows for columns with more nulls than this


# ====== 1) Plan schema ======
class ColumnPlan(BaseModel):
    column: str = Field(description="Exact column name")
    role: Role
    null_strategy: NullStrategy
    fill_label: Optional[str] = Field(default=None, description="Label to use when null_strategy is fill_label")
    reason: str = Field(description="At most 12 words")


class Relationship(BaseModel):
    columns: List[str]
    description: str = Field(description="One short sentence on how these columns relate")


class DataPrepPlan(BaseModel):
    dataset_summary: str = Field(description="One sentence: what this dataset seems to describe")
    columns: List[ColumnPlan]
    relationships: List[Relationship] = Field(default_factory=list)


# ====== 2) What the LLM sees: compact stats per column ======
def _num(v):
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return str(v)


def build_column_stats(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        s = df[col]
        info = {
            "dtype": str(s.dtype),
            "null_pct": round(float(s.isna().mean() * 100), 1),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            info["min"], info["max"] = _num(s.min()), _num(s.max())
        else:
            info["examples"] = [str(v)[:30] for v in s.dropna().astype(str).value_counts().head(3).index]
        lines.append(f"- {col}: {json.dumps(info, ensure_ascii=False)}")
    return f"Rows: {len(df)}\nColumns:\n" + "\n".join(lines)


_PREP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior data engineer preparing a dataset for an AI data analyst. "
     "You only PLAN. Python code will execute your plan.\n\n"
     "Rules:\n"
     "- Return one entry for EVERY column, using the exact column names.\n"
     "- role: identifier (keys / IDs), date, metric (numeric amounts or counts to sum or average), "
     "dimension (categories to group by), free_text (long text or names), "
     "sensitive (personal data), other.\n"
     "- null_strategy, decided from the MEANING of the column and its null percentage:\n"
     "    keep_nan: default for numeric measurements (calculations ignore NaN).\n"
     "    fill_median: numeric only, and only when few values are missing.\n"
     "    fill_zero: numeric, when a null clearly means 'none' (the amount does not apply).\n"
     "    fill_label: text or categorical. Use a label such as 'Not applicable' when a null means the field "
     "does not apply to that row, or 'Unknown' when the value is simply missing.\n"
     "    drop_rows: only for a column with under 1% nulls whose value is essential.\n"
     "- Never drop columns.\n"
     "- Write every text field (summary, reasons, labels, descriptions) in English only.\n"
     "- relationships: groups of columns that are derived from each other, duplicate each other, "
     "or are keys linking entities. At most 6.\n\n"
     "Column statistics:\n{stats}"),
    ("human", "Prepare this dataset."),
])


# ====== 3) Ask the LLM for the plan (with retries and a safe fallback) ======
def _wait_seconds(error) -> float:
    """Read how long Groq asks us to wait after a rate-limit (429) error."""
    m = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(error))
    if not m:
        return 10.0
    return float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1


def plan_data_prep(df: pd.DataFrame, llm, max_retries: int = 4) -> DataPrepPlan:
    chain = _PREP_PROMPT | llm.with_structured_output(DataPrepPlan)
    payload = {"stats": build_column_stats(df)}

    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(payload)
        except RateLimitError as e:  # per-minute token limit: wait, then retry
            wait = _wait_seconds(e)
            print(f"[data prep] rate limit, waiting {wait:.0f}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)
        except BadRequestError:  # model answered in plain text instead of structured output
            print(f"[data prep] attempt {attempt}/{max_retries} failed (no structured output), retrying...")

    print("[data prep] LLM plan unavailable, using safe defaults for every column.")
    return DataPrepPlan(dataset_summary="(fallback: no LLM plan)", columns=[])


# ====== 4) Execute the plan with plain pandas (every decision is guarded) ======
def _resolve_strategy(strategy, label, null_pct, is_numeric, is_datetime):
    """Return (strategy, label, note). Downgrades unsafe or incompatible decisions."""
    numeric_like = is_numeric or is_datetime
    default = "keep_nan" if numeric_like else "fill_label"
    note = None

    if strategy is None:
        return default, label or "Missing", "no plan for this column, default used"

    if strategy == "fill_median" and (not is_numeric or null_pct > MAX_MEDIAN_FILL_PCT):
        strategy, note = default, f"median fill refused ({null_pct:.0f}% nulls or not numeric)"
    elif strategy == "fill_zero" and not is_numeric:
        strategy, note = default, "zero fill refused (not numeric)"
    elif strategy == "fill_label" and numeric_like:
        strategy, note = "keep_nan", "label fill refused (numeric/date column)"
    elif strategy == "drop_rows" and null_pct >= MAX_DROP_ROWS_PCT:
        strategy, note = default, f"row drop refused ({null_pct:.0f}% nulls)"

    return strategy, label or "Missing", note


def apply_prep_plan(df: pd.DataFrame, plan: DataPrepPlan):
    df = df.copy()
    by_col = {c.column: c for c in plan.columns if c.column in df.columns}
    log = {"actions": [], "refused": [], "dates_converted": [], "dates_refused": []}

    # Deterministic hygiene (no decisions needed)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip().replace("", np.nan)

    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = before - len(df)

    empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty)
    log["empty_columns_dropped"] = empty

    # Dates: decided by the LLM, validated by code (text columns only, 90% must parse)
    for col, cp in by_col.items():
        if col not in df.columns or cp.role != "date":
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            log["dates_refused"].append(f"{col}: numeric column")
            continue
        original = df[col].notna().sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(df[col], errors="coerce")
        if original and parsed.notna().sum() / original >= 0.9:
            df[col] = parsed
            log["dates_converted"].append(col)
        else:
            log["dates_refused"].append(f"{col}: less than 90% parsed as dates")

    # Null handling, column by column
    for col in list(df.columns):
        if col not in df.columns:
            continue
        null_pct = float(df[col].isna().mean() * 100)
        if null_pct == 0:
            continue

        cp = by_col.get(col)
        strategy, label, note = _resolve_strategy(
            cp.null_strategy if cp else None,
            cp.fill_label if cp else None,
            null_pct,
            pd.api.types.is_numeric_dtype(df[col]),
            pd.api.types.is_datetime64_any_dtype(df[col]),
        )
        if note:
            log["refused"].append(f"{col}: {note}")

        if strategy == "fill_median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "fill_zero":
            df[col] = df[col].fillna(0)
        elif strategy == "fill_label":
            df[col] = df[col].astype("object").fillna(label)
        elif strategy == "drop_rows":
            df = df[df[col].notna()]

        detail = f" '{label}'" if strategy == "fill_label" else ""
        log["actions"].append(f"{col}: {null_pct:.1f}% nulls -> {strategy}{detail}")

    return df.reset_index(drop=True), log


# ====== 5) Dataset description for the analysis Planner (built from the LLM's roles) ======
def prep_to_text(plan: DataPrepPlan, df: pd.DataFrame, n_rows: int = 2) -> str:
    roles = {c.column: c.role for c in plan.columns if c.column in df.columns}

    def cols(*wanted):
        return [c for c in df.columns if roles.get(c, "other") in wanted]

    hidden = cols("free_text", "sensitive")
    shown = [c for c in df.columns if c not in hidden]

    lines = [
        f"Summary: {plan.dataset_summary}",
        f"Rows: {len(df)}, Columns: {df.shape[1]}",
        f"Date columns: {cols('date')}",
        f"Metric columns (numbers to sum/average): {cols('metric')}",
        f"Dimension columns (categories to group by): {cols('dimension')}",
        f"Identifier columns (never analyze): {cols('identifier')}",
        f"Other columns: {cols('other')}",
        f"(Hidden free-text / personal columns: {hidden})",
    ]
    if plan.relationships:
        lines.append("Relationships between columns:")
        lines += [f"  - {', '.join(r.columns)}: {r.description}" for r in plan.relationships]
    lines.append(f"\nSample rows:\n{df[shown].head(n_rows).to_string()}")
    return "\n".join(lines)