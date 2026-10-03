"""
planner.py
-----------
Analysis Planner: turns a question into an ordered plan of analysis tools.
The LLM only PLANS. Code validates the plan before anything runs.

Guards:
  - retry on rate limit (429) and on invalid tool calls (400)
  - a plan may never touch restricted columns (personal / free-text, decided by the data-prep roles)
"""

import time
from typing import Iterable, List, Literal, Optional

from groq import BadRequestError, RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from consultant import _wait_seconds

TOOLS_DESCRIPTION = """
- descriptive_analysis(column): summary statistics for ONE numeric or categorical column
- group_analysis(metric, group_by, agg): aggregate a numeric column (metric) by a categorical column (group_by); agg is one of sum/mean/count/min/max
- ratio_analysis(numerator, denominator, group_by): ratio of the SUMS of two numeric columns per group. Use it for margins, rates and efficiency, e.g. a margin = profit_column / sales_column per group. Prefer it over averaging a percentage column.
- time_series_analysis(date_column, metric, freq): aggregate a numeric column over time (needs a date column)
- correlation_analysis(): correlation matrix for all numeric columns
- detect_anomalies(column, threshold): find outliers in ONE numeric column
"""

PLAN_COLUMN_FIELDS = ("column", "metric", "group_by", "numerator", "denominator", "date_column")


class PlannerError(Exception):
    """The planner could not produce a valid plan (technical failure, not a refusal)."""


class PlanStep(BaseModel):
    step_id: int = Field(description="Step number starting from 1")
    tool: Literal[
        "descriptive_analysis", "group_analysis", "ratio_analysis",
        "time_series_analysis", "correlation_analysis", "detect_anomalies",
    ] = Field(description="Name of the tool to run")
    column: Optional[str] = Field(default=None, description="Column name (descriptive_analysis, detect_anomalies)")
    metric: Optional[str] = Field(default=None, description="Numeric column to aggregate (group_analysis, time_series_analysis)")
    group_by: Optional[str] = Field(default=None, description="Categorical column to group by (group_analysis, ratio_analysis)")
    agg: Optional[str] = Field(default=None, description="sum, mean, count, min or max (group_analysis)")
    numerator: Optional[str] = Field(default=None, description="Numeric column on top of the ratio (ratio_analysis)")
    denominator: Optional[str] = Field(default=None, description="Numeric column at the bottom of the ratio (ratio_analysis)")
    date_column: Optional[str] = Field(default=None, description="Date column (time_series_analysis)")
    goal: str = Field(description="What this step answers, in one short sentence")


class AnalysisPlan(BaseModel):
    can_answer: bool = Field(description="False if the dataset cannot answer the question")
    reason_if_not: Optional[str] = Field(default=None, description="Why it cannot be answered")
    steps: List[PlanStep] = Field(default_factory=list)


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior data analyst. You only PLAN, you never execute.\n"
     "Write the shortest ordered plan that answers the user's question using ONLY these tools:\n"
     "{tools}\n\n"
     "Rules:\n"
     "- Use ONLY column names that appear in the dataset description. Never invent columns.\n"
     "- Never analyze the columns listed as identifier columns.\n"
     "- When the question or the business goal involves profitability, efficiency or return on spend, "
     "include a ratio_analysis step (a profit column divided by a revenue column) next to the totals, "
     "so groups are compared on efficiency and not only on size.\n"
     "- If the dataset cannot answer the question, set can_answer=false and explain why.\n"
     "- Write every text field in English only.\n\n"
     "Dataset description:\n{profile}"),
    ("human", "{question}"),
])


def plan_analysis(question: str, profile_text: str, llm,
                  sensitive: Iterable[str] = (), max_retries: int = 3) -> AnalysisPlan:
    """Plan an analysis. Raises PlannerError if every attempt fails technically."""
    chain = PLANNER_PROMPT | llm.with_structured_output(AnalysisPlan)
    payload = {"tools": TOOLS_DESCRIPTION, "profile": profile_text, "question": question}

    plan = None
    for attempt in range(1, max_retries + 1):
        try:
            plan = chain.invoke(payload)
            break
        except RateLimitError as e:
            wait = _wait_seconds(e)
            print(f"[planner] rate limit, waiting {wait:.0f}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)
        except BadRequestError:
            print(f"[planner] attempt {attempt}/{max_retries} failed (invalid tool call), retrying...")

    if plan is None:
        raise PlannerError("The planner could not produce a valid plan after several attempts.")

    # Code guard: a plan may never touch restricted columns
    used = {getattr(s, f) for s in plan.steps for f in PLAN_COLUMN_FIELDS if getattr(s, f)}
    blocked = used & set(sensitive)
    if blocked:
        return AnalysisPlan(can_answer=False, steps=[],
                            reason_if_not=f"This needs restricted personal columns: {sorted(blocked)}")
    return plan
