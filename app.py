
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from cache import SemanticCache
from cost_tracker import (
    estimate_tokens,
    calculate_cost,
    format_currency
)
from database import initialize_db, get_cache_count


# -------------------- CONFIGURATION --------------------

load_dotenv(override=True)

st.set_page_config(
    page_title="SmartLLM",
    page_icon="🤖",
    layout="wide"
)

CACHE_THRESHOLD = 0.80

AI_PROVIDER = os.getenv("AI_PROVIDER", "MOCK").strip().upper()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()


# -------------------- INITIALIZE SQLITE --------------------

initialize_db()


# -------------------- PAGE TITLE --------------------

st.title("🤖 SmartLLM")
st.subheader("Semantic Cache and Cost Optimizer")

st.write(
    "Ask questions, reuse similar answers, "
    "and monitor cache performance."
)


# -------------------- SESSION STATE --------------------

if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = SemanticCache(
        threshold=CACHE_THRESHOLD
    )

if "hits" not in st.session_state:
    st.session_state.hits = 0

if "misses" not in st.session_state:
    st.session_state.misses = 0

if "bypasses" not in st.session_state:
    st.session_state.bypasses = 0

if "errors" not in st.session_state:
    st.session_state.errors = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

if "input_tokens" not in st.session_state:
    st.session_state.input_tokens = 0

if "output_tokens" not in st.session_state:
    st.session_state.output_tokens = 0

if "actual_api_cost" not in st.session_state:
    st.session_state.actual_api_cost = 0.0

if "avoided_calls" not in st.session_state:
    st.session_state.avoided_calls = 0

if "estimated_savings" not in st.session_state:
    st.session_state.estimated_savings = 0.0

if "cached_usage" not in st.session_state:
    st.session_state.cached_usage = {}

cache = st.session_state.semantic_cache


# -------------------- MOCK AI PROVIDER --------------------

def generate_mock_answer(question):

    q = question.casefold()

    if "python" in q:
        answer = (
            "Python is a high-level programming language "
            "used for automation, web development, "
            "data science, and AI."
        )

    elif "machine learning" in q:
        answer = (
            "Machine learning is a branch of AI that enables "
            "computers to learn patterns from data."
        )

    elif "semantic cache" in q or "caching" in q:
        answer = (
            "A semantic cache stores answers and reuses them "
            "when a new question is sufficiently similar."
        )

    elif "database" in q:
        answer = (
            "A database is an organized collection of data "
            "that can be stored, retrieved, and managed."
        )

    elif "photosynthesis" in q:
        answer = (
            "Photosynthesis is the process by which plants use "
            "light energy to produce sugars from water and carbon dioxide."
        )

    else:
        answer = (
            f"Simulated response to: {question}\n\n"
            "Mock mode is active. No real AI model was called."
        )

    return (
        "🧪 MOCK MODE — simulated response, not a real LLM answer.\n\n"
        + answer
    )


# -------------------- OPENAI AI PROVIDER --------------------

def generate_openai_answer(question):

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI mode requires an API key. "
            "Set AI_PROVIDER=MOCK in .env to run for free."
        )

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are SmartLLM, a helpful AI assistant. "
            "Answer clearly and accurately. If uncertain, say so. "
            "Do not claim to perform real-world actions."
        ),
        input=question,
        max_output_tokens=400
    )

    answer = response.output_text.strip()

    if not answer:
        raise ValueError("The AI returned an empty answer.")

    usage = response.usage

    return {
        "answer": answer,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens
    }


# -------------------- GENERATE ANSWER --------------------

def generate_answer(question):

    if AI_PROVIDER == "MOCK":

        answer = generate_mock_answer(question)

        return {
            "answer": answer,
            "input_tokens": estimate_tokens(question),
            "output_tokens": estimate_tokens(answer)
        }

    elif AI_PROVIDER == "OPENAI":

        return generate_openai_answer(question)

    else:
        raise ValueError(
            "Invalid AI_PROVIDER. Set it to MOCK or OPENAI in .env."
        )


# -------------------- TRACK GENERATION COST --------------------

def track_generation(usage):

    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]

    request_cost = calculate_cost(
        input_tokens,
        output_tokens
    )

    st.session_state.input_tokens += input_tokens
    st.session_state.output_tokens += output_tokens

    if AI_PROVIDER == "OPENAI":
        st.session_state.api_calls += 1
        st.session_state.actual_api_cost += request_cost

    return request_cost


# -------------------- PROVIDER STATUS --------------------

if AI_PROVIDER == "MOCK":

    st.info("Provider: MOCK — free simulation; no API calls.")

elif AI_PROVIDER == "OPENAI":

    st.info(f"Provider: OPENAI — model: {OPENAI_MODEL}")

else:

    st.error("Invalid AI_PROVIDER setting in .env.")


# -------------------- ASK SMARTLLM --------------------

question = st.text_input("Enter your question")

if st.button("Ask SmartLLM"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        score = None
        answer = None
        result = None

        safety_reason = cache.safety_reason(question)

        try:

            # SAFETY BYPASS

            if safety_reason:

                st.warning(
                    f"Cache bypassed: safety rule "
                    f"'{safety_reason}' matched. "
                    "This question will not be cached."
                )

                with st.spinner("Generating a fresh response..."):
                    usage = generate_answer(question)

                answer = usage["answer"]

                track_generation(usage)

                result = "BYPASS"
                st.session_state.bypasses += 1

            else:

                # CHECK SEMANTIC CACHE

                with st.spinner("Checking semantic cache..."):

                    cached_answer, score = cache.find(question)

                if cached_answer is not None:

                    answer = cached_answer
                    result = "HIT"

                    st.session_state.hits += 1
                    st.session_state.avoided_calls += 1

                    st.success(
                        "Cache HIT — saved answer reused."
                    )

                    # Exact-question savings estimate

                    saved_usage = st.session_state.cached_usage.get(
                        question.casefold().strip()
                    )

                    if saved_usage:
                        st.session_state.estimated_savings += (
                            saved_usage["cost"]
                        )

                else:

                    # GENERATE FRESH ANSWER

                    with st.spinner("Generating a response..."):

                        usage = generate_answer(question)

                    answer = usage["answer"]

                    request_cost = track_generation(usage)

                    # SAVE ANSWER TO SQLITE CACHE

                    saved = cache.save(question, answer)

                    # VERIFY DATABASE SAVE

                    if saved:

                        database_count = get_cache_count()

                        result = "MISS"
                        st.session_state.misses += 1

                        st.session_state.cached_usage[
                            question.casefold().strip()
                        ] = {
                            "input_tokens": usage["input_tokens"],
                            "output_tokens": usage["output_tokens"],
                            "cost": request_cost
                        }

                        st.success(
                            "Cache MISS — response generated and saved."
                        )

                        st.info(
                            f"SQLite verification: {database_count} "
                            "saved entries."
                        )

                    else:

                        result = "BYPASS"
                        st.session_state.bypasses += 1

                        st.warning(
                            "Response generated but not cached."
                        )

        except Exception as exc:

            result = "ERROR"
            st.session_state.errors += 1

            if type(exc).__module__.startswith("openai"):

                st.error(
                    "OpenAI request failed. Check API key, credits, "
                    "model access, and rate limits."
                )

            elif isinstance(exc, ValueError):

                st.error(str(exc))

            else:

                st.error(
                    f"Request failed: {type(exc).__name__}: {exc}"
                )

        # DISPLAY ANSWER

        if answer is not None:

            st.write("### Answer")
            st.write(answer)

        if score is not None:

            st.caption(f"Similarity score: {score:.3f}")

        # SAVE REQUEST HISTORY

        st.session_state.history.append({
            "Question": question,
            "Result": result,
            "Similarity": (
                round(score, 3)
                if score is not None
                else None
            ),
            "Provider": AI_PROVIDER
        })


# -------------------- SQLITE STATUS --------------------

# This runs after question processing, so the count is refreshed.

try:

    current_cache_count = get_cache_count()

    st.caption(
        f"SQLite persistent cache: {current_cache_count} saved entries"
    )

except Exception as exc:

    st.error(f"Could not read SQLite database: {exc}")


# -------------------- PHASE 5 ANALYTICS --------------------

st.divider()

st.header("📊 SmartLLM Cost Optimization Dashboard")

hits = st.session_state.hits
misses = st.session_state.misses
bypasses = st.session_state.bypasses
errors = st.session_state.errors

total = hits + misses

hit_rate = (
    hits / total * 100
    if total
    else 0
)

total_submissions = len(st.session_state.history)


# MAIN METRICS

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Requests", total_submissions)
col2.metric("Cache Hits", hits)
col3.metric("API Calls", st.session_state.api_calls)
col4.metric("Cache Hit Rate", f"{hit_rate:.1f}%")


# TOKEN USAGE

st.subheader("Token Usage")

col5, col6 = st.columns(2)

col5.metric(
    "Input Tokens",
    f"{st.session_state.input_tokens:,}"
)

col6.metric(
    "Output Tokens",
    f"{st.session_state.output_tokens:,}"
)


# COST ANALYSIS

st.subheader("Cost Analysis")

col7, col8 = st.columns(2)

col7.metric(
    "Actual API Cost",
    format_currency(st.session_state.actual_api_cost)
)

col8.metric(
    "Estimated Savings",
    format_currency(st.session_state.estimated_savings)
)

st.caption(
    "MOCK mode uses approximate token estimates and has no paid "
    "API charges. OPENAI mode uses actual API-reported token counts. "
    "Savings are estimates, not billing records."
)


# CACHE PERFORMANCE

st.subheader("Cache Performance")

col9, col10, col11 = st.columns(3)

col9.metric(
    "Avoided Calls",
    st.session_state.avoided_calls
)

col10.metric(
    "Safety Bypasses",
    bypasses
)

col11.metric(
    "Errors",
    errors
)


# -------------------- REQUEST HISTORY --------------------

st.subheader("Request History")

if st.session_state.history:

    df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        df,
        use_container_width=True
    )

else:

    st.write("No requests yet.")


# -------------------- RESET ANALYTICS AND CACHE --------------------

if st.button("Reset Analytics and Cache"):

    # Clear SQLite and in-memory cache
    cache.clear()

    st.session_state.hits = 0
    st.session_state.misses = 0
    st.session_state.bypasses = 0
    st.session_state.errors = 0

    st.session_state.api_calls = 0
    st.session_state.input_tokens = 0
    st.session_state.output_tokens = 0

    st.session_state.actual_api_cost = 0.0
    st.session_state.avoided_calls = 0
    st.session_state.estimated_savings = 0.0

    st.session_state.cached_usage = {}
    st.session_state.history = []

    st.success("Analytics and SQLite cache have been reset.")
    st.rerun()
