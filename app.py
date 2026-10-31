import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from cache import SemanticCache
from cost_tracker import estimate_tokens, calculate_cost, format_currency
from database import initialize_db, get_cache_count

load_dotenv(override=True)

st.set_page_config(
    page_title="SmartLLM | AI Cost Optimizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation callbacks run before Streamlit recreates the radio widget.
def go_to_playground():
    st.session_state["Navigation"] = "AI Playground"


def go_to_history():
    st.session_state["Navigation"] = "Request History"


CACHE_THRESHOLD = 0.80
AI_PROVIDER = os.getenv("AI_PROVIDER", "MOCK").strip().upper()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

initialize_db()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
        --bg:#0b1020; --panel:#121a2b; --panel2:#182238;
        --border:rgba(148,163,184,.15); --text:#eef2ff;
        --muted:#9aa8c2; --purple:#8b78ff; --green:#31c48d;
    }
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
    .stApp{
        background:radial-gradient(circle at 85% 0%,rgba(111,83,255,.12),transparent 28%),
        radial-gradient(circle at 0% 35%,rgba(38,117,160,.08),transparent 25%),var(--bg);
        color:var(--text)
    }
    [data-testid="stHeader"]{background:rgba(11,16,32,.75)}
    [data-testid="stSidebar"]{background:#0e1628;border-right:1px solid var(--border)}
    h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:var(--text)}
    .brand{display:flex;align-items:center;gap:11px;margin:4px 0 26px 2px}
    .brand-icon{width:42px;height:42px;display:flex;align-items:center;justify-content:center;
        border-radius:13px;background:linear-gradient(135deg,#9b87ff,#6652e8);color:white;font-size:22px}
    .brand-name{font-family:'Space Grotesk',sans-serif;font-size:21px;font-weight:700;color:#f5f6ff}
    .brand-caption{color:#8997b2;font-size:10px;letter-spacing:.12em;text-transform:uppercase}
    .eyebrow{color:#a99aff;font-size:11px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px}
    .hero-title{font-family:'Space Grotesk',sans-serif;color:#f4f5ff;font-size:clamp(28px,4vw,39px);
        line-height:1.12;font-weight:700;letter-spacing:-.045em}
    .hero-subtitle{color:var(--muted);font-size:14px;margin-top:10px}
    .status-pill{display:inline-flex;align-items:center;gap:7px;border:1px solid rgba(49,196,141,.25);
        background:rgba(49,196,141,.09);color:#6ee7b7;border-radius:999px;padding:7px 11px;font-size:11px;font-weight:700}
    .status-dot{height:7px;width:7px;background:#31c48d;border-radius:50%}
    .metric-card{background:linear-gradient(145deg,rgba(25,36,59,.98),rgba(18,26,43,.98));
        border:1px solid var(--border);border-radius:17px;padding:18px;min-height:132px}
    .metric-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:15px}
    .metric-label{color:#a4b0c7;font-size:12px;font-weight:600}
    .metric-icon{display:flex;align-items:center;justify-content:center;height:32px;width:32px;
        border-radius:10px;background:rgba(139,120,255,.13);color:#b7aaff;font-size:16px}
    .metric-value{color:#f4f6ff;font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:700}
    .metric-foot{color:#8492ad;font-size:11px;margin-top:3px}
    .section-heading{color:#eef2ff;font-family:'Space Grotesk',sans-serif;font-size:19px;font-weight:600;margin:10px 0 3px}
    .section-caption{color:#91a0ba;font-size:12px;margin-bottom:14px}
    .answer-card{background:linear-gradient(145deg,rgba(25,36,59,.98),rgba(18,26,43,.98));
        border:1px solid var(--border);border-radius:17px;padding:20px;margin-top:14px}
    .answer-label{color:#a99aff;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;margin-bottom:12px}
    .result-hit,.result-miss,.result-bypass{display:inline-flex;padding:6px 10px;border-radius:999px;font-size:11px;font-weight:700}
    .result-hit{background:rgba(49,196,141,.12);border:1px solid rgba(49,196,141,.25);color:#6ee7b7}
    .result-miss{background:rgba(245,166,74,.12);border:1px solid rgba(245,166,74,.25);color:#fdc47e}
    .result-bypass{background:rgba(251,113,133,.12);border:1px solid rgba(251,113,133,.25);color:#fda4af}
    .sidebar-note{color:#7f8da8;font-size:11px;line-height:1.6;padding:12px;background:rgba(255,255,255,.025);
        border:1px solid var(--border);border-radius:12px}
    div[data-testid="stTextInput"] input,div[data-testid="stTextArea"] textarea{
        background:#111b2e!important;color:#f4f6ff!important;border:1px solid rgba(148,163,184,.22)!important;border-radius:12px!important}
    div.stButton>button,div.stFormSubmitButton>button{background:linear-gradient(135deg,#927fff,#6d59ec);color:white;
        border:0;border-radius:11px;font-weight:700;padding:.58rem 1.1rem}
    div.stButton>button:hover,div.stFormSubmitButton>button:hover{border:0;color:white;filter:brightness(1.1)}
    div[data-testid="stMetric"]{background:rgba(25,36,59,.85);border:1px solid var(--border);padding:14px;border-radius:13px}
    div[data-testid="stMetric"] label{color:#9aa8c2!important}
    div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#f4f6ff!important}
    div[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:13px;overflow:hidden}
    hr{border-color:var(--border)}
    .footer{color:#6f7e99;font-size:11px;text-align:center;padding:25px 0 8px}

    /* Premium purple SaaS dashboard layer */
    :root {
        --bg:#090812; --panel:#171321; --panel2:#211a30;
        --border:rgba(190,155,255,.14); --text:#f7f2ff;
        --muted:#a99bbd; --purple:#a33cff; --green:#35d6a0;
    }
    .stApp {
        background:
          radial-gradient(ellipse at 52% -12%,rgba(153,55,255,.23),transparent 38%),
          radial-gradient(ellipse at 100% 72%,rgba(102,32,157,.12),transparent 32%),
          #090812 !important;
    }
    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#17111f 0%,#100d18 100%) !important;
        border-right:1px solid rgba(190,155,255,.12) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius:10px; padding:7px 10px; margin:2px 0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background:rgba(164,66,255,.12);
    }
    .top-gradient {
        height:7px; border-radius:0 0 14px 14px; margin:-1.8rem -1rem 1.6rem;
        background:linear-gradient(90deg,#7b2cff,#c43cff,#792cff);
        box-shadow:0 4px 28px rgba(164,60,255,.38);
    }
    .top-toolbar {
        display:flex; align-items:center; justify-content:space-between; gap:16px;
        padding:12px 17px; margin:0 0 25px; border-radius:15px;
        background:linear-gradient(100deg,rgba(125,47,255,.92),rgba(185,48,230,.92));
        box-shadow:0 8px 32px rgba(139,52,255,.2);
    }
    .toolbar-search {
        color:rgba(255,255,255,.78); font-size:12px; padding:9px 14px;
        width:42%; border-radius:9px; background:rgba(255,255,255,.13);
        border:1px solid rgba(255,255,255,.15);
    }
    .toolbar-meta {color:#fff;font-size:12px;font-weight:600;letter-spacing:.02em}
    .hero-title {font-size:clamp(30px,3.5vw,40px);letter-spacing:-.05em}
    .eyebrow {color:#c28cff}
    .metric-card {
        background:linear-gradient(145deg,rgba(37,29,52,.98),rgba(22,18,32,.98));
        border:1px solid rgba(192,151,255,.16); border-radius:16px;
        box-shadow:0 10px 28px rgba(0,0,0,.16); min-height:143px;
        transition:transform .18s ease,border-color .18s ease;
    }
    .metric-card:hover {transform:translateY(-3px);border-color:rgba(188,113,255,.5)}
    .metric-icon {background:linear-gradient(145deg,rgba(161,61,255,.25),rgba(109,63,209,.16));color:#d4a8ff}
    .metric-value {font-size:29px}
    .section-heading {font-size:18px}
    .answer-card,.sidebar-note {
        background:linear-gradient(145deg,rgba(37,29,52,.98),rgba(22,18,32,.98));
        border-color:rgba(192,151,255,.16);
    }
    div.stButton>button,div.stFormSubmitButton>button,div[data-testid="stDownloadButton"] button {
        background:linear-gradient(110deg,#8c35ff,#bd38df) !important;
        border:1px solid rgba(255,255,255,.12) !important;
        box-shadow:0 5px 18px rgba(147,54,255,.19);
    }
    div.stButton>button:hover,div.stFormSubmitButton>button:hover {
        filter:brightness(1.12);transform:translateY(-1px);
    }
    div[data-testid="stMetric"] {
        background:linear-gradient(145deg,#251c34,#171321);
        border:1px solid rgba(192,151,255,.15);
    }
    div[data-testid="stDataFrame"] {border:1px solid rgba(192,151,255,.15)}
    div[data-testid="stPlotlyChart"],div[data-testid="stVegaLiteChart"] {
        background:linear-gradient(145deg,rgba(37,29,52,.6),rgba(22,18,32,.55));
        border:1px solid rgba(192,151,255,.12);border-radius:15px;padding:8px;
    }
    .activity-row {
        display:flex;align-items:center;gap:11px;padding:11px 0;
        border-bottom:1px solid rgba(192,151,255,.09);font-size:12px;
    }
    .activity-icon {
        width:29px;height:29px;display:flex;align-items:center;justify-content:center;
        border-radius:9px;background:rgba(151,68,255,.17);color:#d4a8ff;flex:none;
    }
    .activity-main {color:#eee5fa;font-weight:600}
    .activity-sub {color:#9689a8;font-size:10px;margin-top:3px}
    .quick-card {
        border:1px solid rgba(192,151,255,.14);border-radius:12px;
        background:linear-gradient(145deg,rgba(37,29,52,.88),rgba(22,18,32,.88));
        text-align:center;padding:16px 8px;color:#e9dcff;font-size:12px;
    }
    .quick-icon {font-size:18px;color:#c48cff;margin-bottom:7px}
    .status-pill {background:rgba(53,214,160,.08)}
    .footer {color:#756982}


    .chart-panel {
        min-height: 255px; padding: 23px 22px;
        border: 1px solid rgba(192,151,255,.15); border-radius: 16px;
        background: linear-gradient(145deg,rgba(37,29,52,.92),rgba(19,15,29,.96));
        display: flex; flex-direction: column; justify-content: center;
        gap: 23px; box-shadow: 0 12px 28px rgba(0,0,0,.14);
    }
    .bar-row { width:100%; }
    .bar-label {
        display:flex; justify-content:space-between; align-items:center;
        gap:12px; color:#cfc2df; font-size:12px; margin-bottom:9px;
    }
    .bar-label b { color:#f7f0ff; font-size:12px; font-weight:700; }
    .bar-track {
        height:9px; width:100%; overflow:hidden; border-radius:20px;
        background:rgba(255,255,255,.075);
    }
    .bar-fill {
        height:100%; min-width:0; border-radius:20px;
        box-shadow:0 0 14px rgba(163,60,255,.2);
        transition:width .35s ease;
    }
    .chart-total {
        display:flex; justify-content:space-between; align-items:center;
        border-top:1px solid rgba(192,151,255,.12); padding-top:16px;
        color:#a99bbd; font-size:11px;
    }
    .chart-total b { color:#f7f0ff; font-size:15px; }

    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULTS = {
    "semantic_cache": None, "hits": 0, "misses": 0, "bypasses": 0,
    "errors": 0, "history": [], "api_calls": 0, "input_tokens": 0,
    "output_tokens": 0, "actual_api_cost": 0.0, "avoided_calls": 0,
    "estimated_savings": 0.0, "cached_usage": {}, "last_result": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.semantic_cache is None:
    st.session_state.semantic_cache = SemanticCache(threshold=CACHE_THRESHOLD)
cache = st.session_state.semantic_cache


def generate_mock_answer(question):
    q = question.casefold()
    if "python" in q:
        answer = "Python is a high-level programming language used for automation, web development, data science, and AI."
    elif "machine learning" in q:
        answer = "Machine learning is a branch of AI that enables computers to learn patterns from data."
    elif "semantic cache" in q or "caching" in q:
        answer = "A semantic cache stores answers and reuses them when a new question is sufficiently similar."
    elif "database" in q:
        answer = "A database is an organized collection of data that can be stored, retrieved, and managed."
    elif "photosynthesis" in q:
        answer = "Photosynthesis is the process by which plants use light energy to produce sugars from water and carbon dioxide."
    else:
        answer = f"Simulated response to: {question}\n\nMock mode is active. No real AI model was called."
    return "🧪 MOCK MODE — simulated response, not a real LLM answer.\n\n" + answer


def generate_openai_answer(question):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI mode requires an API key. Set AI_PROVIDER=MOCK in .env to run for free.")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=("You are SmartLLM, a helpful AI assistant. Answer clearly and accurately. "
                      "If uncertain, say so. Do not claim to perform real-world actions."),
        input=question,
        max_output_tokens=400,
    )
    answer = response.output_text.strip()
    if not answer:
        raise ValueError("The AI returned an empty answer.")
    return {"answer": answer, "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens}


def generate_answer(question):
    if AI_PROVIDER == "MOCK":
        answer = generate_mock_answer(question)
        return {"answer": answer, "input_tokens": estimate_tokens(question),
                "output_tokens": estimate_tokens(answer)}
    if AI_PROVIDER == "OPENAI":
        return generate_openai_answer(question)
    raise ValueError("Invalid AI_PROVIDER. Set it to MOCK or OPENAI in .env.")


def track_generation(usage):
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    request_cost = calculate_cost(input_tokens, output_tokens)
    st.session_state.input_tokens += input_tokens
    st.session_state.output_tokens += output_tokens
    if AI_PROVIDER == "OPENAI":
        st.session_state.api_calls += 1
        st.session_state.actual_api_cost += request_cost
    return request_cost


def process_question(question):
    """Run one prompt through the existing safety, semantic-cache, and provider workflow."""
    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
        return

    score, answer, result_type = None, None, None
    safety_reason = cache.safety_reason(question)
    try:
        if safety_reason:
            with st.spinner("Generating a fresh response..."):
                usage = generate_answer(question)
            answer = usage["answer"]
            track_generation(usage)
            result_type = "BYPASS"
            st.session_state.bypasses += 1
        else:
            with st.spinner("Searching semantic cache..."):
                cached_answer, score = cache.find(question)
            if cached_answer is not None:
                answer, result_type = cached_answer, "HIT"
                st.session_state.hits += 1
                st.session_state.avoided_calls += 1
                saved_usage = st.session_state.cached_usage.get(question.casefold().strip())
                if saved_usage:
                    st.session_state.estimated_savings += saved_usage["cost"]
            else:
                with st.spinner("Generating a fresh response..."):
                    usage = generate_answer(question)
                answer = usage["answer"]
                request_cost = track_generation(usage)
                saved = cache.save(question, answer)
                if saved:
                    result_type = "MISS"
                    st.session_state.misses += 1
                    st.session_state.cached_usage[question.casefold().strip()] = {
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"], "cost": request_cost}
                else:
                    result_type = "BYPASS"
                    st.session_state.bypasses += 1
    except Exception as exc:
        result_type = "ERROR"
        st.session_state.errors += 1
        if type(exc).__module__.startswith("openai"):
            answer = "OpenAI request failed. Check your API key, credits, model access, and rate limits."
        else:
            answer = f"{type(exc).__name__}: {exc}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.last_result = {
        "question": question, "answer": answer, "result": result_type,
        "score": score, "provider": AI_PROVIDER, "timestamp": timestamp,
        "safety_reason": safety_reason}
    st.session_state.history.append({
        "Time": timestamp, "Question": question, "Result": result_type,
        "Similarity": round(score, 3) if score is not None else None,
        "Provider": AI_PROVIDER})


def metric_card(label, value, icon, foot):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-top"><div class="metric-label">{label}</div><div class="metric-icon">{icon}</div></div>
      <div class="metric-value">{value}</div><div class="metric-foot">{foot}</div>
    </div>""", unsafe_allow_html=True)


def page_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="hero-title">{title}</div>'
                f'<div class="hero-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def display_last_result():
    result = st.session_state.last_result
    if not result:
        return
    kind = result["result"]
    if kind == "HIT":
        st.markdown('<span class="result-hit">✓ CACHE HIT · SAVED ANSWER REUSED</span>', unsafe_allow_html=True)
    elif kind == "MISS":
        st.markdown('<span class="result-miss">✦ CACHE MISS · NEW ANSWER GENERATED</span>', unsafe_allow_html=True)
    elif kind == "BYPASS":
        st.markdown('<span class="result-bypass">⚠ SAFETY BYPASS · NOT CACHED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="result-bypass">! REQUEST ERROR</span>', unsafe_allow_html=True)
    if result.get("score") is not None:
        st.caption(f"Semantic similarity score: {result['score']:.3f}")
    st.markdown('<div class="answer-card"><div class="answer-label">SmartLLM response</div>', unsafe_allow_html=True)
    st.write(result["answer"])
    st.markdown("</div>", unsafe_allow_html=True)


# Sidebar and navigation
with st.sidebar:
    st.markdown("""
    <div class="brand"><div class="brand-icon">⚡</div><div>
    <div class="brand-name">SmartLLM</div><div class="brand-caption">Semantic Cache Studio</div>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Overview", "AI Playground", "Request History"],
        label_visibility="collapsed", key="Navigation",
        format_func=lambda x: {"Overview": "◫   Overview", "AI Playground": "✦   AI Playground",
                               "Request History": "▤   Request History"}[x])
    st.divider()
    provider_label = "MOCK · Simulation" if AI_PROVIDER == "MOCK" else (
        f"OPENAI · {OPENAI_MODEL}" if AI_PROVIDER == "OPENAI" else "Provider configuration error")
    st.markdown(f"""<div class="sidebar-note"><b style="color:#e7eaff;">ACTIVE PROVIDER</b><br>
    <span style="color:#a99aff;">●</span> {provider_label}<br><br>
    <b style="color:#e7eaff;">CACHE THRESHOLD</b><br>{CACHE_THRESHOLD:.2f} similarity</div>""",
        unsafe_allow_html=True)
    st.write("")
    if st.button("↻  Reset analytics & cache", use_container_width=True):
        st.session_state.confirm_reset = True
    if st.session_state.get("confirm_reset", False):
        st.warning("This clears the local SQLite cache and resets session analytics.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm", use_container_width=True):
                cache.clear()
                for key, value in {
                    "hits": 0, "misses": 0, "bypasses": 0, "errors": 0,
                    "history": [], "api_calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "actual_api_cost": 0.0, "avoided_calls": 0, "estimated_savings": 0.0,
                    "cached_usage": {}, "last_result": None, "confirm_reset": False,
                }.items():
                    st.session_state[key] = value
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()
    st.markdown('<div class="footer">SMARTLLM 2.0<br>Built for intelligent AI workflows</div>',
                unsafe_allow_html=True)

hits = st.session_state.hits
misses = st.session_state.misses
bypasses = st.session_state.bypasses
errors = st.session_state.errors
cacheable = hits + misses
hit_rate = hits / cacheable * 100 if cacheable else 0.0
try:
    database_count = get_cache_count()
except Exception:
    database_count = len(cache.questions)


# Premium workspace toolbar

st.markdown(
    '<div class="top-gradient"></div>'
    '<div class="top-toolbar"><div class="toolbar-meta">⚡ &nbsp; SMARTLLM WORKSPACE</div>'
    '<div class="toolbar-meta">SEMANTIC CACHE STUDIO &nbsp; · &nbsp; AI COST OPTIMIZER</div></div>',
    unsafe_allow_html=True,
)

# Overview
if page == "Overview":
    page_header("SMARTLLM ANALYTICS", "Dashboard",
        "Welcome back. Here's what's happening across your semantic cache.")
    st.write("")
    status = "MOCK MODE · SIMULATED RESPONSES" if AI_PROVIDER == "MOCK" else (
        f"OPENAI CONNECTED · {OPENAI_MODEL}" if AI_PROVIDER == "OPENAI" else "PROVIDER CONFIGURATION ERROR")
    st.markdown(f'<div class="status-pill"><span class="status-dot"></span>{status}</div>', unsafe_allow_html=True)
    st.write("")

    st.markdown(
        '<div class="section-heading">Ask SmartLLM</div>'
        '<div class="section-caption">Enter a question here. SmartLLM will check the semantic cache before generating a response.</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        with st.form("dashboard_prompt_form", clear_on_submit=True):
            dashboard_question = st.text_area(
                "Ask a question",
                placeholder="Ask SmartLLM anything... e.g. Explain semantic caching in simple terms.",
                height=95,
                label_visibility="collapsed",
            )
            dashboard_submit = st.form_submit_button(
                "✦  Send to SmartLLM", use_container_width=True
            )
    if dashboard_submit:
        if not dashboard_question or not dashboard_question.strip():
            st.warning("Please enter a question.")
        else:
            process_question(dashboard_question)
            st.rerun()

    if st.session_state.last_result:
        latest = st.session_state.last_result
        st.markdown('<div class="section-heading">Latest dashboard response</div>', unsafe_allow_html=True)
        if latest.get("safety_reason"):
            st.warning(f"Safety rule '{latest['safety_reason']}' matched. This question was not cached.")
        display_last_result()
        st.caption(
            f"Question: {latest['question']}  ·  Processed: {latest['timestamp']}  ·  Provider: {latest['provider']}"
        )

    st.write("")
    cols = st.columns(4)
    with cols[0]: metric_card("CACHE HIT RATE", f"{hit_rate:.1f}%", "◉", "Share of cacheable requests")
    with cols[1]: metric_card("SAVED ENTRIES", f"{database_count:,}", "▣", "Persistent SQLite records")
    with cols[2]: metric_card("AVOIDED AI CALLS", f"{st.session_state.avoided_calls:,}", "⚡", "Requests served from cache")
    with cols[3]: metric_card("API SPEND", format_currency(st.session_state.actual_api_cost), "$", "Reported provider cost")

    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(
            '<div class="section-heading">Usage trends</div>'
            '<div class="section-caption">Request volume by outcome in this session</div>',
            unsafe_allow_html=True,
        )
        request_rows = [
            ("Cache hits", hits, "#35d6a0"),
            ("Cache misses", misses, "#a33cff"),
            ("Safety bypasses", bypasses, "#e5a34c"),
            ("Errors", errors, "#ef718b"),
        ]
        request_max = max([value for _, value, _ in request_rows] + [1])
        request_html = '<div class="chart-panel">'
        for label, value, color in request_rows:
            width = value / request_max * 100 if value else 0
            request_html += (
                '<div class="bar-row">'
                f'<div class="bar-label"><span>{label}</span><b>{value:,}</b></div>'
                '<div class="bar-track">'
                f'<div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div>'
                '</div></div>'
            )
        request_html += '</div>'
        st.markdown(request_html, unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="section-heading">Model performance</div>'
            '<div class="section-caption">Token usage by request type</div>',
            unsafe_allow_html=True,
        )
        input_count = st.session_state.input_tokens
        output_count = st.session_state.output_tokens
        token_rows = [
            ("Input tokens", input_count, "#a33cff"),
            ("Output tokens", output_count, "#d16cff"),
        ]
        token_max = max([value for _, value, _ in token_rows] + [1])
        token_html = '<div class="chart-panel">'
        for label, value, color in token_rows:
            width = value / token_max * 100 if value else 0
            token_html += (
                '<div class="bar-row">'
                f'<div class="bar-label"><span>{label}</span><b>{value:,}</b></div>'
                '<div class="bar-track">'
                f'<div class="bar-fill" style="width:{width:.1f}%;background:{color}"></div>'
                '</div></div>'
            )
        token_html += (
            '<div class="chart-total"><span>Total tokens processed</span>'
            f'<b>{input_count + output_count:,}</b></div></div>'
        )
        st.markdown(token_html, unsafe_allow_html=True)

    st.write("")
    left2, right2 = st.columns([1, 1.3])
    with left2:
        st.markdown('<div class="section-heading">Recent activity</div><div class="section-caption">Latest SmartLLM events</div>', unsafe_allow_html=True)
        if st.session_state.history:
            for item in st.session_state.history[-5:][::-1]:
                result = item.get("Result", "REQUEST")
                icon = "✓" if result == "HIT" else ("✦" if result == "MISS" else "!")
                label = "Cache hit" if result == "HIT" else ("New response generated" if result == "MISS" else result.title())
                st.markdown(
                    f'<div class="activity-row"><div class="activity-icon">{icon}</div><div><div class="activity-main">{label}</div><div class="activity-sub">{item.get("Question","")[:58]}</div></div></div>',
                    unsafe_allow_html=True)
        else:
            st.info("No activity yet. Submit your first prompt in AI Playground.")
    with right2:
        st.markdown('<div class="section-heading">Cache performance</div><div class="section-caption">Your semantic cache at a glance</div>', unsafe_allow_html=True)
        performance = [
            ("Cache hits", hits, "#35d6a0"),
            ("Cache misses", misses, "#a33cff"),
            ("Safety bypasses", bypasses, "#e5a34c"),
            ("Errors", errors, "#ef718b"),
        ]
        total_events = max(1, hits + misses + bypasses + errors)
        for label, value, color in performance:
            pct = value / total_events * 100
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;color:#e9def5;font-size:12px;margin:12px 0 5px"><span>{label}</span><b>{value}</b></div>'
                f'<div style="height:7px;border-radius:10px;background:#30263d;overflow:hidden"><div style="height:7px;width:{pct:.1f}%;background:{color};border-radius:10px"></div></div>',
                unsafe_allow_html=True)
        st.write("")
        st.metric("Estimated savings", format_currency(st.session_state.estimated_savings))

    st.write("")
    st.markdown('<div class="section-heading">Quick actions</div><div class="section-caption">Jump straight into your workflow</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown('<div class="quick-card"><div class="quick-icon">✦</div>Generate response</div>', unsafe_allow_html=True)
        st.button("Open AI Playground", key="quick_playground", use_container_width=True,
                  on_click=go_to_playground)
    with q2:
        st.markdown('<div class="quick-card"><div class="quick-icon">▤</div>Review requests</div>', unsafe_allow_html=True)
        st.button("View request history", key="quick_history", use_container_width=True,
                  on_click=go_to_history)
    with q3:
        st.markdown('<div class="quick-card"><div class="quick-icon">⌁</div>Inspect cache</div>', unsafe_allow_html=True)
        st.caption(f"{database_count:,} saved cache entries")
    with q4:
        st.markdown('<div class="quick-card"><div class="quick-icon">↗</div>Cost overview</div>', unsafe_allow_html=True)
        st.caption(f"{format_currency(st.session_state.actual_api_cost)} provider spend")

    st.caption("MOCK mode uses simulated responses and approximate token estimates. No real LLM calls or provider charges occur in MOCK mode. Savings are estimates, not billing records.")

# Playground
elif page == "AI Playground":
    page_header("SMARTLLM ASSISTANT", "Ready to create something new?",
        "Ask anything. SmartLLM checks semantic cache before generating a fresh response.")
    st.write("")
    with st.container(border=True):
        st.markdown("#### ✦ Ask SmartLLM")
        st.caption(f"Provider: {AI_PROVIDER}  ·  Similarity threshold: {CACHE_THRESHOLD:.2f}")
        with st.form("ask_smartllm_form", clear_on_submit=False):
            question = st.text_area("Your question",
                placeholder="For example: Explain machine learning in simple terms...",
                height=120, label_visibility="collapsed")
            submitted = st.form_submit_button("✦  Ask SmartLLM", use_container_width=True)
    if submitted:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            process_question(question)
            st.rerun()
    if st.session_state.last_result:
        st.write("")
        st.markdown('<div class="section-heading">Latest response</div><div class="section-caption">Your most recent SmartLLM result</div>',
                    unsafe_allow_html=True)
        last = st.session_state.last_result
        if last.get("safety_reason"):
            st.warning(f"Safety rule '{last['safety_reason']}' matched. This question was not cached.")
        display_last_result()
        st.caption(f"Question: {last['question']}  ·  Processed: {last['timestamp']}  ·  Provider: {last['provider']}")
    else:
        st.info("Your response will appear here after you submit a question.")


# Request history
elif page == "Request History":
    page_header("ACTIVITY LOG", "Request History",
        "Review questions, cache outcomes, and provider activity from this session.")
    st.write("")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        filter_choice = st.selectbox("Filter requests",
            ["All requests", "Cache HIT", "Cache MISS", "Safety bypass", "Errors"])
        filter_map = {"All requests": None, "Cache HIT": "HIT", "Cache MISS": "MISS",
                      "Safety bypass": "BYPASS", "Errors": "ERROR"}
        selected = filter_map[filter_choice]
        visible_df = history_df if selected is None else history_df[history_df["Result"] == selected]
        st.dataframe(visible_df.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("⬇  Export visible history as CSV",
            data=visible_df.to_csv(index=False).encode("utf-8"),
            file_name="smartllm_request_history.csv", mime="text/csv")
    else:
        st.info("No request history yet. Submit a question in AI Playground.")
    st.write("")
    st.markdown('<div class="section-heading">Session summary</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Total submissions", len(st.session_state.history))
    cols[1].metric("Cache hits", st.session_state.hits)
    cols[2].metric("Cache misses", st.session_state.misses)
    cols[3].metric("Safety bypasses", st.session_state.bypasses)

st.markdown('<div class="footer">SMARTLLM 2.0 · SEMANTIC CACHE & COST OPTIMIZER<br>Intelligent caching for more efficient AI workflows</div>',
            unsafe_allow_html=True)
