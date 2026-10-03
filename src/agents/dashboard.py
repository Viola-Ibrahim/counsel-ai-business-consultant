"""
dashboard.py
-------------
Builds ONE self-contained HTML dashboard from the consultant runs (no Power BI needed).
The user just opens the file in a browser; no internet, no libraries.

Dataset-agnostic and LLM-free: chart types are chosen by CODE from the tool that produced
each step (group/ratio -> bars, time series -> line, correlation -> strongest pairs,
descriptive/anomalies -> table). All text is inserted with textContent, so LLM output
can never inject HTML.
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from consultant import NUM_PATTERN, evidence_is_verified
from powerbi_export import RunRecord, _flatten_output, build_kpis


def _build_data(runs: List[RunRecord], df, metric_columns, title) -> dict:
    runs_data = []
    for run in runs:
        rep = run.report
        steps = {}
        for r in run.results:
            for row in _flatten_output(r["step"], r["output"]):
                if not math.isfinite(row["value"]):
                    continue
                s = steps.setdefault(row["step_id"], {
                    "step_id": row["step_id"], "tool": row["tool"], "goal": row["goal"],
                    "metric": row["metric"], "dimension": row["dimension"], "rows": []})
                s["rows"].append({"label": row["label"], "period": row["period"], "value": row["value"]})

        runs_data.append({
            "question": run.question, "goal": run.goal,
            "situation": rep.situation_summary, "limitations": rep.limitations,
            "findings": [{"text": t,
                          "verified": evidence_is_verified(t, run.results_text) if re.search(NUM_PATTERN, t) else None}
                         for t in rep.key_findings],
            "recs": [{"priority": r.priority, "text": r.recommendation, "evidence": r.evidence,
                      "impact": r.expected_impact, "verified": ok}
                     for r, ok in zip(rep.recommendations, run.verified)],
            "steps": list(steps.values()),
        })

    kpis = []
    if df is not None and len(metric_columns):
        k = build_kpis(df, metric_columns)
        kpis = json.loads(k.to_json(orient="records")) if len(k) else []
    return {"title": title, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "kpis": kpis, "runs": runs_data}


def dashboard_html(runs: List[RunRecord], df: Optional[pd.DataFrame] = None,
                   metric_columns: Sequence[str] = (), title: str = "Business Consultant Dashboard") -> str:
    """The dashboard as one HTML string (for apps that serve or download it)."""
    if not runs:
        raise ValueError("No runs to build a dashboard from (RUNS is empty).")
    data = _build_data(runs, df, metric_columns, title)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", payload).replace("__TITLE__", title.replace("<", "&lt;"))


def build_dashboard(runs: List[RunRecord], out_path, df: Optional[pd.DataFrame] = None,
                    metric_columns: Sequence[str] = (), title: str = "Business Consultant Dashboard") -> Path:
    """Same dashboard, saved to a file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dashboard_html(runs, df, metric_columns, title), encoding="utf-8")
    print(f"Dashboard saved: {out}")
    return out


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1c2430;--muted:#667085;--line:#e4e7ec;--accent:#2f6fed;--bar:#2f6fed;--neg:#d6453d;--ok:#1a8f4c;--warn:#c27a00;}
@media(prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--text:#e8ecf2;--muted:#9aa4b5;--line:#2b3340;--accent:#6b9bff;--bar:#6b9bff;--neg:#ff7a72;--ok:#4cc582;--warn:#f0b24a;}}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.5}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0}h2{font-size:16px;margin:0 0 10px}.muted{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.kpi{border:1px solid var(--line);border-radius:10px;padding:10px 12px}.kpi b{display:block;font-size:12px;color:var(--muted);word-break:break-word}.kpi span{font-size:18px;font-weight:600}
select{width:100%;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--text);font-size:14px}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:99px;border:1px solid currentColor;margin-right:6px}
.High{color:var(--neg)}.Medium{color:var(--warn)}.Low{color:var(--muted)}.ok{color:var(--ok)}.bad{color:var(--neg)}
.rec{padding:10px 0;border-top:1px solid var(--line)}.rec:first-child{border-top:0}
.row{display:grid;grid-template-columns:minmax(80px,200px) 1fr 110px;gap:8px;align-items:center;font-size:13px;margin:4px 0}
.row .l{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.row .v{text-align:right;font-variant-numeric:tabular-nums}
.track{background:var(--line);border-radius:4px;height:14px;overflow:hidden}.fill{height:100%;background:var(--bar)}.fill.neg{background:var(--neg)}
table{border-collapse:collapse;width:100%;font-size:13px}td{padding:4px 8px;border-top:1px solid var(--line)}
svg{width:100%;height:auto}.scroll{overflow-x:auto}
</style></head><body><main>
<h1 id="title"></h1><div class="muted" id="gen"></div>
<div class="card" id="kpiCard" hidden><h2>Dataset overview</h2><div class="grid" id="kpis"></div></div>
<div class="card"><h2>Question</h2><select id="sel"></select><div class="muted" id="goal" style="margin-top:8px"></div></div>
<div id="out"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e;};
const fmt=v=>{const a=Math.abs(v);return a>=1000?v.toLocaleString(undefined,{maximumFractionDigits:0}):a>=10?String(+v.toFixed(1)):String(+v.toFixed(4));};
document.getElementById('title').textContent=D.title;
document.getElementById('gen').textContent='Generated '+D.generated+' | '+D.runs.length+' question(s)';

if(D.kpis.length){document.getElementById('kpiCard').hidden=false;const g=document.getElementById('kpis');
 D.kpis.forEach(k=>{const c=el('div','kpi');c.append(el('b',0,k.metric),el('span',0,'Total '+fmt(k.total)),
  el('div','muted','Mean '+fmt(k.mean)+(k.null_pct>0?' | '+k.null_pct+'% missing':'')));g.append(c);});}

function bars(rows,limit){const box=el('div');const max=Math.max(...rows.map(r=>Math.abs(r.value)))||1;
 rows.slice(0,limit).forEach(r=>{const d=el('div','row');const t=el('div','track'),f=el('div','fill'+(r.value<0?' neg':''));
  f.style.width=(Math.abs(r.value)/max*100)+'%';t.append(f);d.append(el('div','l',r.label),t,el('div','v',fmt(r.value)));box.append(d);});
 if(rows.length>limit)box.append(el('div','muted','Showing '+limit+' of '+rows.length));return box;}

function line(rows){const W=600,H=200,P=30;const v=rows.map(r=>r.value);const lo=Math.min(...v),hi=Math.max(...v),sp=(hi-lo)||1;
 const pts=rows.map((r,i)=>[P+i*(W-2*P)/Math.max(rows.length-1,1),H-P-(r.value-lo)/sp*(H-2*P)]);
 const ns='http://www.w3.org/2000/svg';const s=document.createElementNS(ns,'svg');s.setAttribute('viewBox','0 0 '+W+' '+H);
 const pl=document.createElementNS(ns,'polyline');pl.setAttribute('points',pts.map(p=>p.join(',')).join(' '));
 pl.setAttribute('fill','none');pl.setAttribute('stroke','var(--accent)');pl.setAttribute('stroke-width','2');s.append(pl);
 const tx=(x,y,t,a)=>{const e=document.createElementNS(ns,'text');e.setAttribute('x',x);e.setAttribute('y',y);e.setAttribute('font-size','11');e.setAttribute('fill','var(--muted)');e.setAttribute('text-anchor',a);e.textContent=t;s.append(e);};
 tx(P,H-8,rows[0].label,'start');tx(W-P,H-8,rows[rows.length-1].label,'end');tx(P,12,'max '+fmt(hi),'start');tx(W-P,12,'min '+fmt(lo),'end');
 const w=el('div','scroll');w.append(s);return w;}

function table(rows){const t=el('table');rows.forEach(r=>{const tr=el('tr');tr.append(el('td',0,r.label),el('td',0,fmt(r.value)));t.append(tr);});return t;}

function stepView(s){const c=el('div','card');c.append(el('h2',0,s.goal||('Step '+s.step_id)),
 el('div','muted',s.tool+(s.metric?' | '+s.metric:'')+(s.dimension?' by '+s.dimension:'')));
 const body=el('div');body.style.marginTop='10px';let rows=s.rows;
 if(s.tool==='time_series_analysis'&&rows.length>1)body.append(line(rows));
 else if(s.tool==='correlation_analysis'){rows=[...rows].sort((a,b)=>Math.abs(b.value)-Math.abs(a.value));body.append(bars(rows,10));}
 else if(s.tool==='group_analysis'||s.tool==='ratio_analysis')body.append(bars(rows,15));
 else body.append(table(rows));
 c.append(body);return c;}

function render(i){const r=D.runs[i],out=document.getElementById('out');out.replaceChildren();
 document.getElementById('goal').textContent='Business goal: '+r.goal;
 const sit=el('div','card');sit.append(el('h2',0,'Situation'),el('div',0,r.situation));out.append(sit);
 const f=el('div','card');f.append(el('h2',0,'Key findings'));
 r.findings.forEach(x=>{const d=el('div','rec');if(x.verified===false){d.append(el('span','badge bad','numbers not verified'));}d.append(el('span',0,x.text));f.append(d);});out.append(f);
 const rc=el('div','card');rc.append(el('h2',0,'Recommendations'));
 r.recs.forEach(x=>{const d=el('div','rec');const h=el('div');h.append(el('span','badge '+x.priority,x.priority),
  el('span','badge '+(x.verified?'ok':'bad'),x.verified?'evidence verified':'evidence NOT verified'));d.append(h,el('div',0,x.text),
  el('div','muted','Evidence: '+x.evidence),el('div','muted','Impact: '+x.impact));rc.append(d);});out.append(rc);
 r.steps.forEach(s=>out.append(stepView(s)));
 const l=el('div','card');l.append(el('h2',0,'Limitations'),el('div',0,r.limitations));out.append(l);}

const sel=document.getElementById('sel');
D.runs.forEach((r,i)=>{const o=el('option',0,r.question);o.value=i;sel.append(o);});
sel.addEventListener('change',()=>render(+sel.value));render(0);
</script></body></html>
"""
