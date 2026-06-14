"""
app.py
------
Streamlit frontend for the ReAct Document Analysis System.

Features:
- Drag-and-drop document upload (PDF/DOCX/TXT) -> calls FastAPI /upload
- Chat interface for asking questions -> calls FastAPI /query
- Sidebar showing indexed documents (with delete option) -> /documents
- Real-time status dashboard -> /status
- Expandable ReAct reasoning trace (Thought/Action/Observation) per answer
"""

import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ReAct Document Analysis", layout="wide")
st.title("📄 ReAct Document Analysis System")
st.caption("Ask questions about your documents — powered by a ReAct agent, ChromaDB, and Groq LLM.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_status():
    try:
        r = requests.get(f"{API_URL}/status", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def upload_file(file):
    files = {"file": (file.name, file.getvalue())}
    r = requests.post(f"{API_URL}/upload", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def delete_document(filename):
    r = requests.delete(f"{API_URL}/documents/{filename}", timeout=30)
    r.raise_for_status()
    return r.json()


def ask_question(question, use_decomposition=True):
    payload = {"question": question, "use_decomposition": use_decomposition}
    r = requests.post(f"{API_URL}/query", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def display_step(step):
    """Render a single ReAct trace step (action or final)."""
    if step.get("type") == "action":
        st.markdown(f"**Thought:** {step['thought']}")
        st.markdown(f"**Action:** `{step['action']}` → `{step['action_input']}`")
        st.markdown(f"**Observation:** {step['observation'][:500]}")
        st.divider()
    elif step.get("type") in ("final", "final_forced"):
        st.markdown(f"**Final reasoning:** {step.get('answer', '')[:500]}")


def render_trace(trace):
    """Render a full trace, handling decomposed sub-question traces."""
    for step in trace:
        if "sub_question" in step:
            st.markdown(f"**Sub-question:** {step['sub_question']}")
            for s in step["trace"]:
                display_step(s)
        else:
            display_step(step)


# ---------------------------------------------------------------------------
# Sidebar: status dashboard + document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Status Dashboard")

    status = get_status()
    if "error" in status:
        st.error(f"Backend unreachable: {status['error']}")
    else:
        st.metric("Total Chunks Indexed", status.get("total_chunks", 0))
        st.write(f"**LLM Model:** {status.get('llm_model', '—')}")
        st.write(f"**Embedding Model:** {status.get('embedding_model', '—')}")

    st.divider()
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Drag and drop PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            key = f"uploaded_{file.name}"
            if key not in st.session_state:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        result = upload_file(file)
                        st.success(f"✅ {file.name}: {result['chunks_added']} chunks added")
                        st.session_state[key] = True
                    except Exception as e:
                        st.error(f"❌ Failed to upload {file.name}: {e}")

    st.divider()
    st.header("📚 Indexed Documents")

    docs = status.get("indexed_documents", []) if "error" not in status else []
    if not docs:
        st.info("No documents indexed yet.")
    else:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {doc}")
            if col2.button("🗑️", key=f"del_{doc}"):
                delete_document(doc)
                st.rerun()


# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

use_decomposition = st.toggle("Enable query decomposition (for multi-part questions)", value=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("🔍 Show reasoning trace"):
                render_trace(msg["trace"])

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Running ReAct agent..."):
            try:
                result = ask_question(question, use_decomposition)
                answer = result["answer"]
                trace = result.get("trace", [])

                st.markdown(answer)

                if trace:
                    with st.expander("🔍 Show reasoning trace"):
                        render_trace(trace)

                st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "trace": []})