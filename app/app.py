"""
app.py: Streamlit front-end for the Business Consultant agent.

Run locally:   streamlit run app.py
Each user brings their own Groq key (kept in the browser session only, never stored or logged).
An optional shared key can be set as GROQ_API_KEY in Streamlit secrets (with a small question limit).
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Works with a flat folder or with src/<subfolders> (same idea as the notebook)
ROOT = Path(__file__).resolve().parent
for base in (ROOT, ROOT / "src"):
    if base.exists():
        sys.path.append(str(base))
        sys.path.extend(str(p) for p in base.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))

from consultant import format_report  # noqa: E402
from loader import FileValidationError  # noqa: E402
from plan_cache import PlanCache  # noqa: E402
from session import AnalystSession  # noqa: E402

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_FILE_MB = 20
MAX_Q_OWN_KEY = 30
MAX_Q_SHARED_KEY = 5

st.set_page_config(page_title="Business Consultant", layout="wide")


@st.cache_resource
def get_cache() -> PlanCache:
    return PlanCache(".cache/plans")        # plans only (column names, roles, tool choices): never rows


def make_llms(api_key: str):
    from langchain_groq import ChatGroq
    main = ChatGroq(model=MODEL, api_key=api_key, temperature=0.3, reasoning_effort="medium")
    prep = ChatGroq(model=MODEL, api_key=api_key, temperature=0.2, reasoning_effort="low")
    return main, prep


def shared_key() -> str:
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:                        # no secrets file
        return ""


# ====== Sidebar: key + privacy ======
with st.sidebar:
    st.header("Settings")
    user_key = st.text_input("Your Groq API key", type="password",
                             help="Free keys: console.groq.com. Used only in this browser session.")
    api_key = user_key or shared_key()
    using_own_key = bool(user_key)
    limit = MAX_Q_OWN_KEY if using_own_key else MAX_Q_SHARED_KEY
    if not api_key:
        st.warning("Enter your Groq API key to start.")
    elif not using_own_key:
        st.info(f"Using the shared key: limited to {MAX_Q_SHARED_KEY} questions per session. Add your own key for more.")

    st.subheader("Privacy")
    st.caption(
        "Your file is processed in memory and deleted right after loading. "
        "To plan the analysis, Groq receives column statistics (including a few example values per text column), "
        "two sample rows, and the numbers in the results. Do not upload data you are not allowed to share. "
        "Only plans (column names and tool choices) are cached, never your rows."
    )

st.title("Business Consultant")
st.caption("Upload a dataset, state a business goal, and get evidence-based recommendations.")

if not api_key:
    st.stop()

llm, prep_llm = make_llms(api_key)

# ====== 1) Upload + data prep ======
file = st.file_uploader("Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if file is not None:
    if file.size > MAX_FILE_MB * 1024 * 1024:
        st.error(f"File is too large (max {MAX_FILE_MB} MB).")
        st.stop()

    file_id = (file.name, file.size)
    if st.session_state.get("file_id") != file_id:
        suffix = Path(file.name).suffix.lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            with st.spinner("Reading the data and preparing it..."):
                session = AnalystSession(llm, prep_llm, get_cache(), max_questions=limit)
                st.session_state.info = session.load(tmp_path)
            st.session_state.session = session
            st.session_state.file_id = file_id
            st.session_state.last = None
        except FileValidationError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Could not prepare this file ({type(e).__name__}). Try again or use a different file.")
            st.stop()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)         # the uploaded copy never stays on disk

if "session" not in st.session_state:
    st.stop()

session: AnalystSession = st.session_state.session
session.set_llms(llm, prep_llm)             # follow the key currently in the sidebar
session.max_questions = limit
info = st.session_state.info

st.success(f"{info['summary']}  |  {info['rows']:,} rows, {info['columns']} columns")
if info["from_cache"]:
    st.caption("Data-prep plan loaded from cache (no LLM call).")
if info["restricted_columns"]:
    st.caption(f"{info['restricted_columns']} personal/free-text columns are hidden from the analysis.")
with st.expander("How the data was prepared"):
    st.write("Column roles:", info["roles"])
    st.write("Missing values:", info["null_actions"] or "none")
    if info["refused"]:
        st.write("Decisions blocked by safety guards:", info["refused"])

# ====== 2) Ask ======
st.subheader("Ask a question")
question = st.text_input("Question", placeholder="e.g. Which groups perform best, and how do they compare on efficiency?")
goal = st.text_input("Business goal", placeholder="e.g. Increase overall profitability")
left = session.max_questions - session.n_questions
st.caption(f"{left} question(s) left in this session.")

if st.button("Analyze", type="primary", disabled=not (question.strip() and goal.strip())):
    with st.spinner("Planning, analyzing and writing the report..."):
        st.session_state.last = session.ask(question.strip(), goal.strip())

res = st.session_state.get("last")
if res is not None:
    if res.status == "report":
        st.text(format_report(res.report, res.verified))      # plain text: model output is never rendered as HTML
        if res.plan_from_cache:
            st.caption("Analysis plan loaded from cache.")
    elif res.status == "refused":
        st.warning(f"I can't answer this with the current dataset: {res.message}")
    elif res.status == "consult_failed" and res.fallback_text:
        st.info("The structured report failed, so here is a plain-text answer:")
        st.text(res.fallback_text)
    else:
        st.error({"limit": res.message,
                  "planner_failed": "The planner failed. Please try again in a moment.",
                  "no_results": "No analysis step succeeded for this question.",
                  "consult_failed": "The report could not be generated. Please try again."}.get(res.status, res.message))
    for f in res.failed_steps:
        st.caption(f"Failed: {f}")

# ====== 3) Dashboard ======
if session.runs:
    st.subheader("Dashboard")
    html = session.dashboard()
    st.download_button("Download dashboard (HTML)", html, file_name="dashboard.html", mime="text/html")
    with st.expander("Preview", expanded=False):
        components.html(html, height=900, scrolling=True)
