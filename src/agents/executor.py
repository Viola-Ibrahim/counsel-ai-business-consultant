"""
executor.py
------------
Runs an AnalysisPlan step by step with the real analysis functions.
A failed step is recorded instead of crashing the whole run.
"""

from analysis import (
    correlation_analysis,
    descriptive_analysis,
    detect_anomalies,
    group_analysis,
    ratio_analysis,
    time_series_analysis,
)


def run_step(step, df):
    """Run ONE plan step and return its raw result."""
    if step.tool == "descriptive_analysis":
        return descriptive_analysis(df, step.column)
    if step.tool == "group_analysis":
        return group_analysis(df, step.metric, step.group_by, step.agg or "sum")
    if step.tool == "ratio_analysis":
        return ratio_analysis(df, step.numerator, step.denominator, step.group_by)
    if step.tool == "time_series_analysis":
        return time_series_analysis(df, step.date_column, step.metric, freq="ME")
    if step.tool == "correlation_analysis":
        return correlation_analysis(df)
    if step.tool == "detect_anomalies":
        return detect_anomalies(df, step.column)
    raise ValueError(f"Unknown tool: {step.tool}")


def execute_plan(plan, df) -> list:
    """Returns [{"step", "output", "error"}, ...]. Empty list if the plan says it cannot answer."""
    if not plan.can_answer:
        return []

    results = []
    for step in plan.steps:
        try:
            results.append({"step": step, "output": run_step(step, df), "error": None})
        except Exception as e:
            results.append({"step": step, "output": None, "error": str(e)})
    return results
