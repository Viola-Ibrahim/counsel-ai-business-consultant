"""
insight.py
-----------
Turns raw tool outputs into short text summaries (done in CODE, so the LLM never
has to judge sizes or gaps), and writes a plain-text answer when the structured
consultant report fails.
"""

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data analyst explaining results to a non-technical user.\n"
     "Rules:\n"
     "- Use ONLY the numbers in the analysis results below. Never invent numbers.\n"
     "- Answer the user's question directly first, then add 2-3 short key insights.\n"
     "- If a step has an ERROR, mention it briefly instead of guessing.\n"
     "- Keep it short and clear.\n"
     "- Answer in English only.\n\n"
     "Analysis results:\n{results}"),
    ("human", "{question}"),
])


def summarize_output(step, output, data) -> str:
    """Turn a raw tool output into a short, unambiguous text summary."""

    # Correlation: send only the top 10 strongest pairs, not the full matrix
    if step.tool == "correlation_analysis":
        id_cols = [c for c in output.columns if str(c).lower().endswith("id")]
        corr = output.drop(index=id_cols, columns=id_cols, errors="ignore")
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)  # upper triangle only
        pairs = corr.where(mask).stack()
        pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(10)
        lines = [f"{a} <-> {b}: {v:.3f}" for (a, b), v in pairs.items()]
        return "Top 10 strongest correlations (by absolute value):\n" + "\n".join(lines)

    # Anomalies: the returned rows ARE the outliers, so say it explicitly
    if step.tool == "detect_anomalies":
        col = step.column
        n_flagged = len(output)
        if n_flagged == 0:
            return f"No anomalies were detected in column '{col}'."
        return (
            f"{n_flagged} anomalous rows (outliers) were detected in '{col}' "
            f"out of {len(data)} total rows.\n"
            f"The outlier values range from {output[col].min()} to {output[col].max()}.\n"
            f"For comparison, the whole column ranges from {data[col].min()} to {data[col].max()} "
            f"with a median of {data[col].median()}."
        )

    # Group / ratio analysis: let the CODE decide if the gap between groups is meaningful
    if step.tool in ("group_analysis", "ratio_analysis") and isinstance(output, pd.Series) and len(output) > 1:
        text = output.to_string()
        lo, hi = float(output.min()), float(output.max())
        if lo > 0:
            gap = (hi - lo) / lo * 100
            verdict = (
                "NEGLIGIBLE (under 5%): treat the groups as practically equal and do NOT rank them."
                if gap < 5 else "meaningful."
            )
            text += (
                f"\nGap between the highest group ({output.idxmax()}) and the lowest group "
                f"({output.idxmin()}): {gap:.1f}% -> {verdict}"
            )
        return text

    return output.to_string() if hasattr(output, "to_string") else str(output)


def results_to_text(results: list, df) -> str:
    """Build the text block the LLM reads (and the numbers are verified against)."""
    parts = []
    for r in results:
        step = r["step"]
        parts.append(f"Step {step.step_id} ({step.tool}) - Goal: {step.goal}")
        if r["error"]:
            parts.append(f"ERROR: {r['error']}")
        else:
            parts.append(summarize_output(step, r["output"], df))
        parts.append("")
    return "\n".join(parts)


def generate_insights(question: str, results: list, df, llm) -> str:
    """Plain-text answer (fallback when the structured consultant report fails)."""
    response = (INSIGHT_PROMPT | llm).invoke({"results": results_to_text(results, df), "question": question})
    if not response.content or not response.content.strip():
        return "(The model returned an empty response. Try running the question again.)"
    return response.content
