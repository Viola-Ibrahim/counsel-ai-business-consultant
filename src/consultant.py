"""
consultant.py
--------------
Business Consultant agent.

Turns analysis results (already summarized to text by the pipeline) into a structured
business report: situation, findings, prioritized recommendations with numeric evidence.

Design notes:
  - The LLM is passed IN (no globals), so this file does not depend on the notebook.
  - The input is `results_text` (a string). Building it from raw results stays in the pipeline.
  - Code verifies that every number cited as evidence exists in the results.
    This checks the NUMBER, not the soundness of the conclusion.
"""

import re
import time
from typing import List, Literal, Tuple

from groq import BadRequestError, RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ====== 1) Shape of the consultant's output ======
class Recommendation(BaseModel):
    recommendation: str = Field(description="One concrete, actionable recommendation")
    priority: Literal["High", "Medium", "Low"] = Field(description="How urgent/important it is for the business goal")
    evidence: str = Field(description="The exact numbers copied from the analysis results that support it")
    step_id: int = Field(description="The analysis step number the evidence comes from")
    expected_impact: str = Field(description="What improvement this could bring, described qualitatively (no invented numbers)")


class ConsultantReport(BaseModel):
    situation_summary: str = Field(description="2-3 sentences: where the business stands based on the data")
    key_findings: List[str] = Field(description="2-4 findings, each based on numbers in the results")
    recommendations: List[Recommendation] = Field(description="2-4 recommendations, highest priority first")
    limitations: str = Field(description="What this data cannot tell us, in one sentence")


# ====== 2) Prompt ======
CONSULTANT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior business consultant. You turn analysis results into decisions.\n"
     "Business goal: {goal}\n\n"
     "Rules:\n"
     "- Use ONLY numbers that appear in the analysis results below. Never invent or compute new numbers.\n"
     "- Every recommendation must cite its evidence (exact numbers) and the step it came from.\n"
     "- Mention ONLY dimensions that appear in the analysis results.\n"
     "- If a step says the gap between groups is NEGLIGIBLE, say those groups perform about the same "
     "and do NOT recommend anything based on that difference.\n"
     "- Totals show SIZE, not efficiency. Do not recommend increasing spend because a group has the "
     "highest total. Recommend spend changes only if the results contain a per-unit efficiency measure "
     "with a meaningful gap; otherwise recommend a test, or name the extra data needed (cost, ROI).\n"
     "- Recommend business actions only. If a step has an ERROR, never turn it into a recommendation.\n"
     "- Prioritize recommendations by importance for the business goal.\n"
     "- In 'limitations', describe only what THESE results cannot show.\n"
     "- Write every text field in English only.\n"
     "- You MUST answer by calling the provided function with the structured report. "
     "Never answer in plain text.\n\n"
     "Analysis results:\n{results}"),
    ("human", "{question}"),
])


def build_consultant_chain(llm):
    """prompt -> LLM -> structured ConsultantReport."""
    return CONSULTANT_PROMPT | llm.with_structured_output(ConsultantReport)


# ====== 3) Code-based check: are the cited numbers really in the results? ======
NUM_PATTERN = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_STEP_REF = re.compile(r"\(?\bstep\s*\d+\)?", re.IGNORECASE)  # "(Step 3)" is a reference, not evidence


def extract_numbers(text: str) -> List[float]:
    return [float(n.replace(",", "")) for n in re.findall(NUM_PATTERN, text)]


def evidence_is_verified(evidence: str, results_text: str) -> bool:
    """
    True only if the evidence contains at least one number AND every number
    appears in the results (allowing for rounding to the decimals that were written).
    """
    cleaned = _STEP_REF.sub("", evidence)
    raw_numbers = re.findall(NUM_PATTERN, cleaned)
    if not raw_numbers:  # evidence with no numbers proves nothing
        return False

    source_numbers = extract_numbers(results_text)
    for raw in raw_numbers:
        n = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(x - n) <= tolerance for x in source_numbers):
            return False
    return True


# ====== 4) Run the consultant (retries on rate limit and on missing structured output) ======
def _wait_seconds(error) -> float:
    """Read how long Groq asks us to wait after a rate-limit (429) error."""
    m = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(error))
    if not m:
        return 10.0
    return float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1


def consult(question: str, goal: str, results_text: str, llm,
            max_retries: int = 3) -> Tuple[ConsultantReport, List[bool]]:
    """
    Returns (report, verified) where verified[i] says whether the evidence of
    recommendation i was found in the results.
    Raises BadRequestError / RateLimitError if every attempt fails.
    """
    chain = build_consultant_chain(llm)
    payload = {"goal": goal, "results": results_text, "question": question}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            report = chain.invoke(payload)
            verified = [evidence_is_verified(r.evidence, results_text) for r in report.recommendations]
            return report, verified
        except RateLimitError as e:
            last_error = e
            wait = _wait_seconds(e)
            print(f"[consultant] rate limit, waiting {wait:.0f}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)
        except BadRequestError as e:
            last_error = e
            print(f"[consultant] attempt {attempt}/{max_retries} failed (no structured output), retrying...")
    raise last_error


# ====== 5) Pretty printing ======
def format_report(report: ConsultantReport, verified: List[bool]) -> str:
    lines = [f"SITUATION: {report.situation_summary}", "", "KEY FINDINGS:"]
    lines += [f" - {f}" for f in report.key_findings]
    lines += ["", "RECOMMENDATIONS:"]
    for rec, ok in zip(report.recommendations, verified):
        flag = "numbers found in results" if ok else "WARNING: evidence not verified in results"
        lines += [
            f"[{rec.priority}] {rec.recommendation}",
            f"    Evidence (step {rec.step_id}): {rec.evidence}   ({flag})",
            f"    Impact: {rec.expected_impact}",
            "",
        ]
    lines.append(f"LIMITATIONS: {report.limitations}")
    return "\n".join(lines)