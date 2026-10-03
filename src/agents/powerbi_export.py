"""
powerbi_export.py
------------------
Exports consultant runs to CSV tables that Power BI can load directly.

Dataset-agnostic: nothing here knows any column name. Tables are built from the
steps' own outputs and from the metric columns chosen by the data-prep layer.

Tables (all linked by run_id, except kpis):
  runs.csv              one row per question asked (the "dimension" table)
  findings.csv          key findings per run (+ whether their numbers exist in the results)
  recommendations.csv   recommendations per run (+ priority, evidence, verified flag)
  analysis_results.csv  every step's output in LONG format (label, value) -> easy to chart
  kpis.csv              totals / averages / null % for each metric column of the dataset

CSVs are written as utf-8-sig so Power BI and Excel read English and Arabic text correctly.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from consultant import NUM_PATTERN, evidence_is_verified


# ====== 1) One record per consultant run ======
@dataclass
class RunRecord:
    question: str
    goal: str
    results: list            # [{"step", "output", "error"}, ...] (successful steps only)
    results_text: str        # the text the consultant saw (used to verify numbers)
    report: Any              # ConsultantReport
    verified: List[bool]     # one flag per recommendation
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ====== 2) Flatten a step output into long rows ======
def _step_labels(step) -> Dict[str, str]:
    """Generic description of what a step measured, from the step's own fields."""
    g = lambda name: getattr(step, name, None)
    if g("numerator") and g("denominator"):
        metric = f"{g('numerator')} / {g('denominator')}"
    else:
        metric = g("metric") or g("column") or ""
    dimension = g("group_by") or g("date_column") or ""
    return {"metric": metric, "dimension": dimension}


def _flatten_output(step, output) -> List[dict]:
    base = {"step_id": getattr(step, "step_id", None), "tool": step.tool,
            "goal": getattr(step, "goal", ""), **_step_labels(step)}
    rows = []

    def add(label, value, period=None):
        num = pd.to_numeric(value, errors="coerce")
        if pd.notna(num):
            rows.append({**base, "label": str(label), "period": period, "value": float(num)})

    if isinstance(output, pd.Series):
        is_time = isinstance(output.index, pd.DatetimeIndex)
        for idx, val in output.items():
            add(idx.strftime("%Y-%m") if is_time else idx, val,
                period=idx.strftime("%Y-%m-%d") if is_time else None)

    elif isinstance(output, dict):  # descriptive statistics
        for key, val in output.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    add(f"{key}: {k2}", v2)
            else:
                add(key, val)

    elif isinstance(output, pd.DataFrame):
        if list(output.index) == list(output.columns):  # correlation matrix -> unique pairs
            cols = list(output.columns)
            for i, a in enumerate(cols):
                for b in cols[i + 1:]:
                    add(f"{a} <-> {b}", output.loc[a, b])
        else:  # e.g. anomalies: keep only the count, never raw rows
            add("rows_returned", len(output))

    return rows


# ====== 3) Dataset-level KPIs (from metric columns chosen by the data-prep layer) ======
def build_kpis(df: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in metric_columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col]
        rows.append({
            "metric": col,
            "rows_with_value": int(s.notna().sum()),
            "null_pct": round(float(s.isna().mean() * 100), 2),
            "total": float(s.sum()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)


# ====== 4) Export ======
def export_runs(runs: List[RunRecord], out_dir, df: Optional[pd.DataFrame] = None,
                metric_columns: Sequence[str] = ()) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs_rows, finding_rows, rec_rows, result_rows = [], [], [], []

    for i, run in enumerate(runs, 1):
        run_id = f"R{i:03d}"
        rep = run.report

        runs_rows.append({
            "run_id": run_id, "timestamp": run.timestamp,
            "question": run.question, "goal": run.goal,
            "n_steps": len(run.results),
            "n_recommendations": len(rep.recommendations),
            "n_recs_verified": int(sum(run.verified)),
            "situation_summary": rep.situation_summary,
            "limitations": rep.limitations,
        })

        for n, text in enumerate(rep.key_findings, 1):
            has_numbers = bool(re.search(NUM_PATTERN, text))
            finding_rows.append({
                "run_id": run_id, "finding_no": n, "finding": text,
                "numbers_verified": evidence_is_verified(text, run.results_text) if has_numbers else None,
            })

        for n, (rec, ok) in enumerate(zip(rep.recommendations, run.verified), 1):
            rec_rows.append({
                "run_id": run_id, "rec_no": n, "priority": rec.priority,
                "priority_rank": {"High": 1, "Medium": 2, "Low": 3}.get(rec.priority, 9),  # for sorting in Power BI
                "recommendation": rec.recommendation, "evidence": rec.evidence,
                "step_id": rec.step_id, "expected_impact": rec.expected_impact,
                "evidence_verified": ok,
            })

        for r in run.results:
            for row in _flatten_output(r["step"], r["output"]):
                result_rows.append({"run_id": run_id, **row})

    tables = {
        "runs": pd.DataFrame(runs_rows),
        "findings": pd.DataFrame(finding_rows),
        "recommendations": pd.DataFrame(rec_rows),
        "analysis_results": pd.DataFrame(result_rows),
    }
    if df is not None and len(metric_columns):
        tables["kpis"] = build_kpis(df, metric_columns)

    paths = {}
    for name, table in tables.items():
        path = out / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
        print(f"{name}.csv: {len(table)} rows")
    return paths