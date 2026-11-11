import os
import tempfile
import time

import streamlit as st
from google import genai
from google.genai import errors

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- config ----
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📚", layout="centered")

# Tried in order. If one returns a 4xx (bad ID, quota), the next is used.
MODEL_CHOICES = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("Add GEMINI_API_KEY in Streamlit Cloud → App settings → Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)


# --------------------------------------------------------------- helpers ----
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_index(files):
    """Load every uploaded PDF, chunk it, and build an in-memory FAISS index."""
    all_docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        try:
            pages = PyPDFLoader(path).load()
        finally:
            os.remove(path)
        for p in pages:
            p.metadata["source"] = f.name
            p.metadata["page"] = int(p.metadata.get("page", 0)) + 1
        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [c for c in splitter.split_documents(all_docs) if c.page_content.strip()]
    if not chunks:
        return None, 0
    store = FAISS.from_documents(chunks, get_embeddings())
    return store, len(chunks)


def ask_gemini(prompt: str, preferred: str) -> tuple[str, str]:
    """Call Gemini with retries and model fallback. Returns (answer, model_used)."""
    order = [preferred] + [m for m in MODEL_CHOICES if m != preferred]
    last_err = None
    for model in order:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                return (resp.text or "(Empty response from model.)"), model
            except errors.ServerError as e:  # 5xx / overloaded: retry w/ backoff
                last_err = e
                time.sleep(2**attempt)
            except errors.ClientError as e:  # 4xx: bad model ID, quota, key...
                last_err = e
                st.toast(f"{model}: {e.code} {e.message}", icon="⚠️")
                break
    raise last_err


def build_prompt(question: str, context: str, history: list[dict]) -> str:
    recent = history[-6:]
    convo = "\n".join(f"{m['role'].title()}: {m['content']}" for m in recent)
    return f"""You are a helpful assistant answering questions about the user's PDF documents.
Use ONLY the context below. If the answer is not in the context, say you don't know.
Be concise and accurate.

Context:
{context}

Recent conversation:
{convo or "(none)"}

Question: {question}"""


# ------------------------------------------------------------------- UI -----
st.title("📚 PDF Q&A Assistant")
st.caption("Upload one or more PDFs, then chat with them.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", MODEL_CHOICES, index=0)
    top_k = st.slider("Chunks to retrieve", 2, 8, 4)
    show_sources = st.checkbox("Show sources", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if not files:
    st.info("Upload a PDF to get started.")
    st.stop()

# Rebuild the index only when the set of files changes (not on every rerun).
file_key = tuple((f.name, f.size) for f in files)
if st.session_state.get("file_key") != file_key:
    with st.spinner("Reading and indexing your PDFs..."):
        store, n_chunks = build_index(files)
    if store is None:
        st.error(
            "No extractable text found. The PDF may be scanned images; "
            "this app needs text-based PDFs."
        )
        st.stop()
    st.session_state.store = store
    st.session_state.file_key = file_key
    st.session_state.messages = []
    st.success(f"Indexed {len(files)} file(s) into {n_chunks} chunks.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['file']}, p.{s['page']}**\n\n> {s['text']}")

# New question
if question := st.chat_input("Ask a question about your PDFs"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            hits = st.session_state.store.similarity_search(question, k=top_k)
            context = "\n\n".join(
                f"[{d.metadata['source']} p.{d.metadata['page']}]\n{d.page_content}"
                for d in hits
            )
            prompt = build_prompt(question, context, st.session_state.messages[:-1])
            try:
                answer, used = ask_gemini(prompt, model_choice)
            except errors.APIError as e:
                answer, used = f"⚠️ Gemini API error {e.code}: {e.message}", None

        st.markdown(answer)
        sources = [
            {
                "file": d.metadata["source"],
                "page": d.metadata["page"],
                "text": d.page_content[:300].replace("\n", " ") + "...",
            }
            for d in hits
        ]
        if show_sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['file']}, p.{s['page']}**\n\n> {s['text']}")
        if used:
            st.caption(f"Answered with `{used}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
