import os, streamlit as st

st.set_page_config(
    page_title="Ganesh Chandrasekaran (Bot) - Rowan University",
    page_icon=":robot_face:"
)

from openai import OpenAI
from databricks.vector_search.client import VectorSearchClient

# Consider reading from environment or Streamlit secrets
DATABRICKS_HOST = st.secrets.get("DATABRICKS_HOST") 
DATABRICKS_TOKEN = st.secrets.get("DATABRICKS_TOKEN")
VS_ENDPOINT = "prof-vs-endpoint"
VS_INDEX = "workspace.ganesh_rowan_bot.docs_index"
LLM_MODEL = "databricks-meta-llama-3-1-405b-instruct"

CUSTOM_CSS = """
    <style>
    :root {
        --bg: #531914;
        --bg-elevated: #6d241d;
        --accent: #f4b183;
        --accent-hover: #e3955d;
        --text-main: #fdf7f2;
        --text-muted: #e0c9c0;
    }

    /* App background and base font */
    .stApp {
        background-color: var(--bg);
        color: var(--text-main);
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Headings and body text */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-main);
    }
    p, label, span, .stMarkdown {
        color: var(--text-main);
    }

    /* Text inputs and textareas */
    .stTextInput > div > div > input,
    textarea {
        background-color: var(--bg-elevated);
        color: var(--text-main);
        border: 1px solid var(--accent);
    }
    .stTextInput > div > div > input::placeholder,
    textarea::placeholder {
        color: var(--text-muted);
    }
    .stTextInput > div > div > input:focus,
    textarea:focus {
        outline: none;
        border-color: var(--accent-hover);
        box-shadow: 0 0 0 1px var(--accent-hover);
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--accent);
        color: #000000;
        border-radius: 999px;
        border: none;
        padding: 0.4rem 1.2rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: var(--accent-hover);
    }
    </style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Ganesh Chandrasekaran (Bot) - Rowan University")


def clear_question():
    st.session_state["question_box"] = ""

def _normalize_vs_results(res):
    # Handle common response shapes; return list[dict]
    if isinstance(res, dict) and isinstance(res.get("data"), list):
        return res["data"]

    if isinstance(res, dict) and "result" in res:
        manifest = res.get("manifest") or {}
        cols_obj = manifest.get("columns") or manifest.get("column_names") or []
        cols = [c["name"] if isinstance(c, dict) and "name" in c else str(c) for c in cols_obj]
        result = res.get("result") or {}
        data_array = result.get("data_array") or result.get("data")
        if data_array and cols:
            rows = []
            for arr in data_array:
                rows.append({cols[i]: arr[i] for i in range(min(len(cols), len(arr)))})
            return rows
        if result.get("row_count", 0) == 0:
            return []
    return []

question = st.text_area("Ask a question about the courses, syllabus, grading, assignments, final project, ai policy, tools to learn.", key="question_box")

col_ask, col_clear = st.columns([3, 1])

with col_ask:
    ask_clicked = st.button("Ask the Bot", key="ask_btn", use_container_width=True)

with col_clear:
    st.button("Clear Text", 
        key="clear_btn", 
        on_click=clear_question,
        use_container_width=True)

if ask_clicked and st.session_state.question_box.strip():
    try:
        with st.spinner("Searching and answering..."):
            # Vector Search
            vsc = VectorSearchClient(
                workspace_url=DATABRICKS_HOST,
                personal_access_token=DATABRICKS_TOKEN,
                disable_notice=True
            )
            index = vsc.get_index(VS_ENDPOINT, VS_INDEX)
            res = index.similarity_search(
                query_text=st.session_state.question_box,
                columns=["title","url_or_path","chunk_text","section","page","doc_id"],
                num_results=5
            )
            rows = _normalize_vs_results(res)
            if not rows:
                st.warning("No matching passages yet. Try rephrasing your question or reach out to Professor Ganesh Chandra via email.")
                st.stop()

            # Build context
            contexts = []
            for row in rows:
                title = row.get("title") or ""
                url = row.get("url_or_path") or ""
                page = row.get("page")
                cite = f"{title} — {url}".strip(" —")
                if page:
                    cite += f" (p.{page})"
                chunk = row.get("chunk_text") or ""
                contexts.append(f"[Source] {cite}\n{chunk}")
            rag_context = "\n\n".join(contexts)

            # Foundation Model APIs via OpenAI client (Databricks-compatible)
            # Use the workspace base URL with /serving-endpoints and the foundation model endpoint name
            client = OpenAI(
                api_key=DATABRICKS_TOKEN,
                base_url=f"{DATABRICKS_HOST.rstrip('/')}/serving-endpoints"
            )
            prompt = (
                "Answer using ONLY the context. Cite sources inline (Title). "
                "If unsure, say you don't know.\n\n"
                f"Question: {question}\n\nContext:\n{rag_context}"
            )
            out = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.2
            )

        # Show the result in the UI (do not use print)
        st.markdown("### Answer")
        st.write(out.choices[0].message.content)

        # Optionally show sources used
        st.markdown("### Sources")
        for row in rows:
            t = row.get("title") or ""
            # u = row.get("url_or_path") or ""
            if t:
                st.markdown(f"- [{t}]")

    except Exception as e:
        st.exception(e)