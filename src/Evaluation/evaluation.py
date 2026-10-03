"""
evaluation.py
--------------
Fixed-question evaluation for the Business Consultant pipeline.

Runs a fixed set of questions through: plan -> execute -> consult, and records
for every question what happened, using CODE checks only (no LLM judging):

  - Did the planner pick the expected tools?
  - Did it refuse when it should (missing data / sensitive columns)?
  - Did it reference columns that do not exist?
  - Did any step fail?
  - How many recommendations have numbers verified in the results?
  - How many numbers in the key findings are NOT found in the results?

The pipeline pieces are passed in as functions, so this file does not depend on the notebook.
Output: a DataFrame (and results.csv) you can rerun after every change and compare.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd

try:
    from groq import RateLimitError
except ImportError:  # keeps the file importable without groq installed
    class RateLimitError(Exception):
        pass

from consultant import evidence_is_verified, _wait_seconds

COLUMN_FIELDS = ("column", "metric", "group_by", "numerator", "denominator", "date_column")


# ====== 1) Test cases ======
@dataclass
class EvalCase:
    id: str
    question: str
    goal: str
    must_refuse: bool = False                    # the dataset cannot answer this
    expected_all: Tuple[str, ...] = ()           # ALL of these tools must be in the plan
    expected_any: Tuple[str, ...] = ()           # at least ONE of these tools must be in the plan
    forbidden_columns: Tuple[str, ...] = ()      # the plan must NOT use these (sensitive data)
    note: str = ""


def load_cases(path: str) -> List[EvalCase]:
    """
    Load test cases from a JSON file (a list of objects with the EvalCase fields).
    Cases are DATA about one dataset, so they live outside this file: the evaluation
    code itself knows nothing about any specific dataset.
    """
    import json
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    cases = []
    for item in raw:
        for key in ("expected_all", "expected_any", "forbidden_columns"):
            item[key] = tuple(item.get(key, ()))
        cases.append(EvalCase(**item))
    return cases


# ====== 2) Helpers ======
def _with_retry(fn: Callable, max_attempts: int = 3):
    """Call fn(); on a Groq rate limit wait the time Groq asks for, then retry."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except RateLimitError as e:
            if attempt == max_attempts:
                raise
            wait = _wait_seconds(e)
            print(f"  [eval] rate limit, waiting {wait:.0f}s...")
            time.sleep(wait)


def _plan_info(plan) -> Tuple[List[str], List[str]]:
    """(tools used, columns referenced) from a plan."""
    tools, cols = [], []
    for step in plan.steps:
        tools.append(step.tool)
        for field in COLUMN_FIELDS:
            value = getattr(step, field, None)
            if value:
                cols.append(value)
    return tools, cols


def _unverified_finding_numbers(report, results_text: str) -> int:
    """How many key findings contain a number that is not in the results."""
    import re
    from consultant import NUM_PATTERN
    count = 0
    for text in report.key_findings:
        if re.search(NUM_PATTERN, text) and not evidence_is_verified(text, results_text):
            count += 1
    return count


# ====== 3) Run one case ======
def run_case(case: EvalCase, df_columns: List[str],
             plan_fn: Callable, execute_fn: Callable,
             results_to_text_fn: Callable, consult_fn: Callable,
             sensitive_columns: Tuple[str, ...] = ()) -> dict:
    row = {
        "id": case.id, "question": case.question, "must_refuse": case.must_refuse,
        "can_answer": None, "tools_used": "", "tools_ok": None,
        "invented_columns": "", "forbidden_used": "", "steps_failed": 0,
        "status": "", "n_recs": 0, "recs_unverified": 0,
        "findings_unverified": 0, "negligible_flagged": None,
        "seconds": 0.0, "passed": False, "error": "", "note": case.note,
    }
    start = time.time()
    forbidden = set(case.forbidden_columns) | set(sensitive_columns)

    try:
        plan = _with_retry(lambda: plan_fn(f"{case.question}\n(Business goal: {case.goal})"))
        row["can_answer"] = bool(plan.can_answer)

        tools, cols = _plan_info(plan)
        row["tools_used"] = ", ".join(tools)
        row["invented_columns"] = ", ".join(sorted({c for c in cols if c not in df_columns}))
        row["forbidden_used"] = ", ".join(sorted({c for c in cols if c in forbidden}))
        row["tools_ok"] = (
            all(t in tools for t in case.expected_all)
            and (not case.expected_any or any(t in tools for t in case.expected_any))
        )

        if not plan.can_answer:
            row["status"] = "refused"
        else:
            results = execute_fn(plan)
            failed = [r for r in results if r["error"]]
            ok = [r for r in results if not r["error"]]
            row["steps_failed"] = len(failed)

            if not ok:
                row["status"] = "no_results"
            else:
                results_text = results_to_text_fn(ok)
                row["negligible_flagged"] = "NEGLIGIBLE" in results_text
                try:
                    report, verified = _with_retry(lambda: consult_fn(case.question, case.goal, results_text))
                    row["status"] = "report"
                    row["n_recs"] = len(report.recommendations)
                    row["recs_unverified"] = verified.count(False)
                    row["findings_unverified"] = _unverified_finding_numbers(report, results_text)
                except Exception as e:  # structured output failed after all retries
                    row["status"] = "consult_failed"
                    row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {str(e)[:120]}"

    # ---- pass / fail rule ----
    if row["status"] == "error" or row["forbidden_used"]:
        row["passed"] = False                                           # sensitive column used = always a fail
    elif case.forbidden_columns:
        row["passed"] = True                                            # explicit privacy test: refusing or avoiding is fine
    elif case.must_refuse:
        row["passed"] = row["can_answer"] is False                      # must say "cannot answer"
    else:
        row["passed"] = bool(
            row["can_answer"] and row["tools_ok"] and not row["invented_columns"]
            and row["steps_failed"] == 0 and row["status"] == "report"
        )

    row["seconds"] = round(time.time() - start, 1)
    return row


# ====== 4) Run everything ======
def run_evaluation(df_columns: List[str],
                   plan_fn: Callable, execute_fn: Callable,
                   results_to_text_fn: Callable, consult_fn: Callable,
                   cases: List[EvalCase],
                   sensitive_columns: Tuple[str, ...] = (),
                   pause_seconds: float = 5.0,
                   save_path: Optional[str] = "results.csv") -> pd.DataFrame:
    """
    plan_fn(question_with_goal) -> AnalysisPlan
    execute_fn(plan)            -> list of {"step", "output", "error"}
    results_to_text_fn(results) -> str
    consult_fn(question, goal, results_text) -> (report, verified)
    cases: from load_cases(...) (dataset-specific data, kept outside the code)
    sensitive_columns: columns the planner must never use (take them from the data-prep roles)
    """
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id} ...")
        rows.append(run_case(case, list(df_columns), plan_fn, execute_fn, results_to_text_fn, consult_fn,
                             sensitive_columns))
        if i < len(cases):
            time.sleep(pause_seconds)  # stay under the per-minute token limit

    results = pd.DataFrame(rows)
    if save_path:
        results.to_csv(save_path, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel/Power BI read it cleanly
    print_summary(results)
    return results


def print_summary(results: pd.DataFrame) -> None:
    cols = ["id", "passed", "status", "tools_used", "recs_unverified", "findings_unverified", "seconds"]
    print("\n" + results[cols].to_string(index=False))

    n, passed = len(results), int(results["passed"].sum())
    reports = results[results["status"] == "report"]
    print(f"\nPassed: {passed}/{n}")
    if len(reports):
        recs = int(reports["n_recs"].sum())
        bad_recs = int(reports["recs_unverified"].sum())
        print(f"Recommendations with verified numbers: {recs - bad_recs}/{recs}")
        print(f"Key findings with numbers not found in results: {int(reports['findings_unverified'].sum())}")

    failed = results[~results["passed"]]
    if len(failed):
        print("\nFailed cases:")
        for _, r in failed.iterrows():
            why = r["error"] or (
                f"can_answer={r['can_answer']}, tools_ok={r['tools_ok']}, "
                f"invented=[{r['invented_columns']}], forbidden=[{r['forbidden_used']}], "
                f"failed_steps={r['steps_failed']}, status={r['status']}"
            )
            print(f"  - {r['id']}: {why}")