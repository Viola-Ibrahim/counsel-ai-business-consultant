"""
session.py
-----------
AnalystSession: everything ONE user needs, in one object (no global variables),
so several users can use the app at the same time without mixing their data.

    s = AnalystSession(llm, prep_llm, cache)
    info = s.load("file.csv")                  # load + LLM data prep (cached per schema)
    result = s.ask("question", "business goal")  # plan (cached) -> execute -> consult
    html = s.dashboard()
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from consultant import consult
from dashboard import dashboard_html
from data_prep import DataPrepPlan, apply_prep_plan, plan_data_prep, prep_to_text
from executor import execute_plan
from insight import generate_insights, results_to_text
from loader import load_file
from plan_cache import PlanCache, make_key, schema_signature
from planner import AnalysisPlan, PlannerError, plan_analysis
from powerbi_export import RunRecord


@dataclass
class AskResult:
    status: str                       # report | refused | limit | planner_failed | no_results | consult_failed
    message: str = ""
    report: Any = None
    verified: List[bool] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    plan_from_cache: bool = False
    fallback_text: str = ""


class AnalystSession:
    def __init__(self, llm, prep_llm=None, cache: Optional[PlanCache] = None, max_questions: int = 20):
        self.llm = llm                          # planner + consultant
        self.prep_llm = prep_llm or llm         # data prep (a lighter model is enough)
        self.cache = cache or PlanCache()
        self.max_questions = max_questions
        self.n_questions = 0
        self.df = None
        self.runs: List[RunRecord] = []

    def set_llms(self, llm, prep_llm=None):
        self.llm, self.prep_llm = llm, prep_llm or llm

    # ---------- 1) load + data prep ----------
    def load(self, file_path: str) -> dict:
        df_raw = load_file(file_path)
        self.signature = schema_signature(df_raw)

        key = make_key("prep", self.signature)
        plan = self.cache.get(key, DataPrepPlan)
        self.prep_from_cache = plan is not None
        if plan is None:
            plan = plan_data_prep(df_raw, self.prep_llm)
            if plan.columns:                    # never cache the empty fallback plan
                self.cache.put(key, plan)

        self.df, self.prep_log = apply_prep_plan(df_raw, plan)   # guards run on the REAL data every time
        self.prep_plan = plan
        self.sensitive = [c.column for c in plan.columns if c.role in ("sensitive", "free_text")]
        self.metric_columns = [c.column for c in plan.columns if c.role == "metric"]
        self.profile_text = prep_to_text(plan, self.df)
        self.runs, self.n_questions = [], 0

        return {
            "summary": plan.dataset_summary,
            "rows": len(self.df), "columns": self.df.shape[1],
            "from_cache": self.prep_from_cache,
            "roles": {c.column: c.role for c in plan.columns},
            "null_actions": self.prep_log["actions"],
            "refused": self.prep_log["refused"],
            "restricted_columns": len(self.sensitive),
        }

    # ---------- 2) ask ----------
    def ask(self, question: str, goal: str) -> AskResult:
        if self.df is None:
            raise RuntimeError("Load a dataset first.")
        if self.n_questions >= self.max_questions:
            return AskResult("limit", f"Question limit reached ({self.max_questions} per session).")
        self.n_questions += 1

        full_question = f"{question}\n(Business goal: {goal})"

        # plan: from cache if the same schema + question + goal was planned before
        key = make_key("plan", self.signature, full_question)
        plan = self.cache.get(key, AnalysisPlan)
        from_cache = plan is not None
        if plan is None:
            try:
                plan = plan_analysis(full_question, self.profile_text, self.llm, self.sensitive)
            except PlannerError as e:
                return AskResult("planner_failed", str(e))
            if not plan.can_answer or plan.steps:   # never cache an empty "answerable" plan
                self.cache.put(key, plan)

        if not plan.can_answer:
            return AskResult("refused", plan.reason_if_not or "The dataset cannot answer this.", plan_from_cache=from_cache)

        results = execute_plan(plan, self.df)
        failed = [f"step {r['step'].step_id} ({r['step'].tool}): {r['error']}" for r in results if r["error"]]
        ok = [r for r in results if not r["error"]]
        if not ok:
            return AskResult("no_results", "No analysis step succeeded.", failed_steps=failed, plan_from_cache=from_cache)

        results_text = results_to_text(ok, self.df)
        try:
            report, verified = consult(question, goal, results_text, self.llm)
        except Exception as e:                      # structured report failed after all retries
            try:
                text = generate_insights(question, ok, self.df, self.llm)
            except Exception:
                text = ""
            return AskResult("consult_failed", f"{type(e).__name__}", failed_steps=failed,
                             plan_from_cache=from_cache, fallback_text=text)

        self.runs.append(RunRecord(question=question, goal=goal, results=ok,
                                   results_text=results_text, report=report, verified=verified))
        return AskResult("report", report=report, verified=verified, failed_steps=failed, plan_from_cache=from_cache)

    # ---------- 3) dashboard ----------
    def dashboard(self, title: str = "Business Consultant Dashboard") -> str:
        return dashboard_html(self.runs, self.df, self.metric_columns, title)
