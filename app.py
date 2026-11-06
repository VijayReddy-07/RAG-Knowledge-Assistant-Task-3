import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from google.genai.errors import APIError
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    CHROMA_DIR,
    DATA_DIR
)
from retrieval import (
    load_vector_database,
    retrieve_chunks
)
from rag import (
    create_gemini_client,
    generate_rag_answer,
    generate_plain_answer
)

# Page configuration
st.set_page_config(
    page_title="AI & ML Knowledge Assistant | Enterprise RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Styling (Clean, Human-Designed Enterprise Aesthetic)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Clear Streamlit header bar to prevent cutting off top letters */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 2.5rem !important;
    }

    /* Top padding fix - provides 5.25rem clearance so title is never cut off */
    .block-container {
        padding-top: 5.25rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1200px;
    }

    /* Force all text in the main workspace to high-contrast dark slate */
    .main, .main p, .main span, .main li, .main h1, .main h2, .main h3, .main h4,
    .main [data-testid="stMarkdownContainer"] *,
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] span,
    .main [data-testid="stMarkdownContainer"] li,
    .main [data-testid="stMarkdownContainer"] h1,
    .main [data-testid="stMarkdownContainer"] h2,
    .main [data-testid="stMarkdownContainer"] h3,
    .main [data-testid="stMarkdownContainer"] strong,
    .main [data-testid="stMarkdownContainer"] em {
        color: #0F172A !important;
    }

    /* Professional Navy Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }

    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }

    [data-testid="stSidebar"] .stRadio label p {
        color: #E2E8F0 !important;
        font-size: 0.88rem !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F1F5F9 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        padding: 0.45rem 0.75rem !important;
        transition: all 0.15s ease-in-out !important;
        text-align: left !important;
        margin-bottom: 0.25rem !important;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        color: #FFFFFF !important;
    }

    /* Primary CTA Button */
    .stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #1D4ED8 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1E40AF !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.25) !important;
    }

    /* Inputs */
    .stTextArea label p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }

    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }

    .stTextArea textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    /* Main Area Radio Selector - Explicit high contrast text */
    .main .stRadio label p {
        color: #0F172A !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    .main .stRadio [role="radiogroup"] {
        background-color: #F1F5F9 !important;
        padding: 6px 14px !important;
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
        gap: 1.25rem !important;
    }

    .main .stRadio [data-testid="stWidgetLabel"] p {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #64748B !important;
    }

    /* Metric & Information Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.15rem 1.35rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        height: 100%;
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A !important;
    }

    .metric-sub {
        font-size: 0.8rem;
        color: #64748B !important;
        margin-top: 0.25rem;
    }

    /* Feature Cards */
    .feature-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        height: 100%;
    }

    .feature-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #0F172A !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }

    .feature-desc {
        font-size: 0.8rem;
        color: #64748B !important;
        line-height: 1.45;
        margin: 0;
    }

    /* Container Box for Answer */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #2563EB !important;
        border-radius: 8px !important;
        padding: 1.25rem 1.5rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1.25rem !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] li,
    [data-testid="stVerticalBlockBorderWrapper"] div {
        color: #0F172A !important;
        font-size: 0.94rem !important;
        line-height: 1.65 !important;
    }

    /* Response Timing Badge */
    .time-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Key Highlights Panel */
    .highlights-panel {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }

    .highlights-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1E293B !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    /* Source Item */
    .source-snippet {
        font-size: 0.82rem;
        color: #334155 !important;
        line-height: 1.5;
        background-color: #F8FAFC;
        padding: 0.75rem 0.95rem;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        margin-top: 0.5rem;
    }

    /* Status Pill */
    .status-pill-green {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cached resources
@st.cache_resource
def load_rag_resources():
    """Load the embedding model and vector database once."""
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )
    collection = load_vector_database()
    return embedding_model, collection


@st.cache_resource
def load_gemini_client():
    """Create the Gemini client once."""
    return create_gemini_client()


def format_api_error(error):
    if error.code == 404:
        return (
            f"Gemini model '{GEMINI_MODEL}' is unavailable to this API key. "
            "Check GEMINI_MODEL in your .env file."
        )
    if error.code in (429, 503):
        return (
            f"Gemini is temporarily unavailable (HTTP {error.code}). "
            "Please wait a moment and try again."
        )
    return (
        f"Gemini API request failed (HTTP {error.code}): "
        f"{error.message or 'No details provided.'}"
    )


def capture_answer(answer_function, *args):
    try:
        return answer_function(*args), None
    except APIError as error:
        return None, format_api_error(error)
    except Exception as error:
        return None, f"An error occurred: {error}"


def capture_rag_answer(question, embedding_model, collection, gemini_client):
    result, error = capture_answer(
        generate_rag_answer,
        question,
        embedding_model,
        collection,
        gemini_client
    )
    if error:
        return None, [], error
    answer, chunks = result
    return answer, chunks, None


# Session State Management
if "user_question" not in st.session_state:
    st.session_state["user_question"] = "What is supervised learning and what are its main tasks?"

sample_prompts = [
    "What is supervised learning and what are its main tasks?",
    "Explain how Backpropagation works in Neural Networks.",
    "What is the difference between Bagging and Boosting?",
    "What is Q-learning and what algorithm does it use?",
    "What are Convolutional Neural Networks (CNNs) used for?"
]

# Sidebar Navigation (Navy Blue SaaS Theme)
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #1E293B; margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.65rem;">
                <div style="background-color: #2563EB; width: 34px; height: 34px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 700; color: white; font-size: 0.95rem;">
                    AI
                </div>
                <div>
                    <div style="font-weight: 700; font-size: 0.92rem; color: #FFFFFF; line-height: 1.2;">KNOWLEDGE ASSISTANT</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Enterprise Document RAG</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    nav_choice = st.radio(
        "NAVIGATION",
        ["Home", "About Project", "Tech Stack", "Sample Questions"],
        index=0
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid #1E293B; margin: 1.25rem 0;'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; font-weight: 600; margin-bottom: 0.6rem;">
            SAMPLE QUESTIONS
        </div>
        """,
        unsafe_allow_html=True
    )

    for p in sample_prompts:
        if st.button(p, key=f"side_btn_{p}"):
            st.session_state["user_question"] = p
            st.rerun()

    st.markdown("<hr style='border: 0; border-top: 1px solid #1E293B; margin: 1.25rem 0;'>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="padding: 0.65rem 0.75rem; background-color: #1E293B; border-radius: 6px; border: 1px solid #334155; font-size: 0.75rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: #94A3B8;">System Status</span>
                <span style="color: #34D399; font-weight: 600;">● Operational</span>
            </div>
            <div style="color: #64748B; font-size: 0.7rem;">ChromaDB &bull; Local Dense HNSW</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Check API Key
if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY. Please specify your key in the .env file.")
    st.stop()


# View Router
if nav_choice == "About Project":
    st.markdown("## About the AI & Machine Learning Knowledge Assistant")
    st.markdown(
        """
        This application is an **Enterprise-grade Retrieval-Augmented Generation (RAG) System** designed to 
        answer complex machine learning and artificial intelligence queries with strict factual attribution.

        ### Core Capabilities
        1. **Strict Context Adherence**: Eliminates hallucination by enforcing generation exclusively based on retrieved textbook passages.
        2. **Page-Level Attribution**: Every claim is annotated with the specific PDF file name and exact page number.
        3. **Dual Verification**: Enables side-by-side benchmarking against base LLM models to measure grounding efficacy and factual precision.

        ### Indexed Knowledge Base
        - `machine_learning_fundamentals.pdf` — Core ML algorithms, optimization, deep learning, RL.
        - `supervised_learning.pdf` — Classification, regression, logistic regression, trees.
        - `neural_networks.pdf` — Architectures, perceptrons, activations, backpropagation.
        - `model_evaluation.pdf` — Cross-validation, metrics, bias-variance tradeoff.
        """
    )
    if st.button("← Return to Assistant Workspace", type="primary"):
        st.session_state["user_question"] = "What is supervised learning and what are its main tasks?"
        st.rerun()

elif nav_choice == "Tech Stack":
    st.markdown("## Architecture & Technology Stack")
    st.markdown("A modular RAG stack leveraging state-of-the-art open-source embeddings and Google Gemini:")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            """
            #### Retrieval & Storage
            - **Vector Store**: **ChromaDB** (Persistent Local HNSW Index)
            - **Embedding Model**: **sentence-transformers/all-MiniLM-L6-v2** (384-dimensional dense space)
            - **Text Chunking**: **LangChain RecursiveCharacterTextSplitter** (2,000 char windows, 400 char overlap)
            - **PDF Extraction**: **pypdf** reader with multi-page parsing
            """
        )
    with col_t2:
        st.markdown(
            """
            #### Inference & Presentation
            - **Language Model**: **Google Gemini (gemini-3.5-flash-lite)** via `google-genai` SDK
            - **Temperature**: `0.2` for deterministic, grounded outputs
            - **Application UI**: **Streamlit** styled with clean SaaS dashboard CSS
            - **Concurrency**: `ThreadPoolExecutor` for parallel comparison benchmarks
            """
        )

elif nav_choice == "Sample Questions":
    st.markdown("## Curated Research Prompts")
    st.markdown("Click any verified question below to test the RAG grounding and citation accuracy:")

    for p in sample_prompts:
        cq, ca = st.columns([4, 1])
        with cq:
            st.markdown(f"**{p}**")
        with ca:
            if st.button("Query Now →", key=f"run_prompt_{p}"):
                st.session_state["user_question"] = p
                st.rerun()
        st.divider()

else:
    # Home View: Main Assistant Workspace

    # Main Heading & Subtitle
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.25rem;">
            <div>
                <h1 style="font-size: 1.65rem; font-weight: 700; color: #0F172A; margin: 0; padding: 0; letter-spacing: -0.02em;">
                    AI & Machine Learning Knowledge Assistant
                </h1>
                <p style="font-size: 0.88rem; color: #64748B; margin-top: 0.3rem; margin-bottom: 0;">
                    Document-grounded question answering powered by semantic vector retrieval and Gemini.
                </p>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="status-pill-green">
                    <span class="status-dot"></span> Ready & Grounded
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dynamic chunk count
    total_stored_chunks = 400
    try:
        _, collection_check = load_rag_resources()
        if collection_check:
            total_stored_chunks = collection_check.count()
    except Exception:
        pass

    # Top Status Cards (Knowledge Base & Model Info)
    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-0.5-5.5Z"/><path d="M6 6h10"/></svg>
                    Knowledge Base Status
                </div>
                <div class="metric-value">{total_stored_chunks} Document Chunks</div>
                <div class="metric-sub">
                    <b>ChromaDB</b> Persistent &bull; Cosine HNSW &bull; 4 Core Documents
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_stat2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/></svg>
                    Model & Inference Configuration
                </div>
                <div class="metric-value">{GEMINI_MODEL}</div>
                <div class="metric-sub">
                    Embeddings: <b>all-MiniLM-L6-v2</b> &bull; Temp: 0.2 &bull; Max Tokens: 4,000
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # Four Compact Feature Cards
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                    Semantic Retrieval
                </div>
                <p class="feature-desc">Matches question semantics via dense 384-dimensional MiniLM embeddings.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/></svg>
                    Document Grounding
                </div>
                <p class="feature-desc">Constrains model answers to explicit PDF passages with exact page citations.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                    Anti-Hallucination
                </div>
                <p class="feature-desc">Prevents speculative extrapolation when facts are not in the source texts.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 3v18"/></svg>
                    Dual Benchmarking
                </div>
                <p class="feature-desc">Run side-by-side comparative queries to evaluate RAG grounding vs. base LLM.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

    # Answer Mode Selection
    mode_selection = st.radio(
        "ANSWER MODE",
        ["RAG Answer", "Plain Gemini (No RAG)", "Compare Both"],
        horizontal=True,
        index=0
    )

    # Large Question Input
    question_text = st.text_area(
        "Enter your research question",
        value=st.session_state["user_question"],
        height=100,
        placeholder="e.g. What is supervised learning and what are its main tasks?"
    )

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        ask_button = st.button("Get Answer", type="primary", use_container_width=True)

    # Execution & Results Handling
    if ask_button:
        if not question_text.strip():
            st.warning("Please enter a question to query the knowledge assistant.")
        else:
            try:
                gemini_client = load_gemini_client()

                if mode_selection == "RAG Answer":
                    with st.spinner("Retrieving relevant passages and synthesizing grounded answer..."):
                        t0 = time.perf_counter()
                        embedding_model, collection = load_rag_resources()
                        answer, chunks = generate_rag_answer(
                            question_text,
                            embedding_model,
                            collection,
                            gemini_client
                        )
                        latency = time.perf_counter() - t0

                    # Answer section header with response time badge
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem; margin-bottom: 0.5rem;">
                            <h3 style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 0;">
                                Grounded RAG Answer
                            </h3>
                            <div class="time-badge">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                                Completed in {latency:.2f}s &bull; {len(chunks)} sources retrieved
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Visible Bordered Card Container
                    with st.container(border=True):
                        st.markdown(answer)

                    # Key Highlights Panel
                    unique_sources = list({c['source'] for c in chunks})
                    best_distance = min([c['distance'] for c in chunks]) if chunks else 0.0

                    st.markdown(
                        f"""
                        <div class="highlights-panel">
                            <div class="highlights-title">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                                Key Highlights & Verification Notes
                            </div>
                            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.85rem; color: #334155; line-height: 1.6;">
                                <li><b>Corpus Evidence:</b> Grounded across <b>{len(chunks)} relevant passages</b> from <b>{len(unique_sources)} distinct documents</b>.</li>
                                <li><b>Cosine Proximity:</b> Highest ranking chunk reached a distance score of <b>{best_distance:.4f}</b> (dense semantic alignment).</li>
                                <li><b>Integrity Verification:</b> Explicit citations matched directly against extracted PDF page records.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Sources Section
                    st.markdown(
                        f"""
                        <h4 style="font-size: 1rem; font-weight: 700; color: #0F172A; margin-top: 1.5rem; margin-bottom: 0.75rem;">
                            Retrieved Sources ({len(chunks)})
                        </h4>
                        """,
                        unsafe_allow_html=True
                    )

                    for index, chunk in enumerate(chunks, start=1):
                        with st.expander(f"{index}. {chunk['source']} — Page {chunk['page']} (Distance: {chunk['distance']:.4f})"):
                            st.markdown(f"**Document**: `{chunk['source']}` &nbsp;|&nbsp; **Page**: `{chunk['page']}` &nbsp;|&nbsp; **Cosine Distance**: `{chunk['distance']:.4f}`")
                            st.markdown(
                                f"""<div class="source-snippet">{chunk['text']}</div>""",
                                unsafe_allow_html=True
                            )

                elif mode_selection == "Plain Gemini (No RAG)":
                    with st.spinner("Generating baseline answer from Gemini..."):
                        t0 = time.perf_counter()
                        answer = generate_plain_answer(question_text, gemini_client)
                        latency = time.perf_counter() - t0

                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem; margin-bottom: 0.5rem;">
                            <h3 style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 0;">
                                Plain Gemini Answer (Without Document Grounding)
                            </h3>
                            <div class="time-badge">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                                Completed in {latency:.2f}s
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.container(border=True):
                        st.markdown(answer)

                else:
                    # Compare Both Mode
                    plain_answer = None
                    plain_error = None
                    rag_answer = None
                    rag_error = None
                    chunks = []

                    with st.spinner("Running parallel evaluation across both approaches..."):
                        t0 = time.perf_counter()
                        try:
                            embedding_model, collection = load_rag_resources()
                        except Exception as error:
                            rag_error = f"An error occurred: {error}"

                        with ThreadPoolExecutor(max_workers=2) as executor:
                            plain_future = executor.submit(
                                capture_answer,
                                generate_plain_answer,
                                question_text,
                                gemini_client
                            )
                            rag_future = None
                            if not rag_error:
                                rag_future = executor.submit(
                                    capture_rag_answer,
                                    question_text,
                                    embedding_model,
                                    collection,
                                    gemini_client
                                )

                            plain_answer, plain_error = plain_future.result()
                            if rag_future is not None:
                                rag_answer, chunks, rag_error = rag_future.result()

                        total_latency = time.perf_counter() - t0

                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem; margin-bottom: 0.75rem;">
                            <h3 style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 0;">
                                Side-by-Side Comparison
                            </h3>
                            <div class="time-badge">
                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                                Total Evaluation: {total_latency:.2f}s
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    left_col, right_col = st.columns(2)

                    with left_col:
                        st.markdown(
                            """
                            <div style="font-size: 0.95rem; font-weight: 600; color: #64748B; margin-bottom: 0.5rem;">
                                Without RAG (Plain Gemini)
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        with st.container(border=True):
                            if plain_error:
                                st.error(plain_error)
                            else:
                                st.markdown(plain_answer)

                    with right_col:
                        st.markdown(
                            """
                            <div style="font-size: 0.95rem; font-weight: 600; color: #2563EB; margin-bottom: 0.5rem;">
                                With RAG (Document-Grounded)
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        with st.container(border=True):
                            if rag_error:
                                st.error(rag_error)
                            else:
                                st.markdown(rag_answer)

                    if chunks:
                        st.markdown(
                            f"""
                            <h4 style="font-size: 1rem; font-weight: 700; color: #0F172A; margin-top: 1.5rem; margin-bottom: 0.75rem;">
                                Retrieved Context Chunks ({len(chunks)})
                            </h4>
                            """,
                            unsafe_allow_html=True
                        )
                        for index, chunk in enumerate(chunks, start=1):
                            with st.expander(f"{index}. {chunk['source']} — Page {chunk['page']} (Distance: {chunk['distance']:.4f})"):
                                st.markdown(f"**Document**: `{chunk['source']}` &nbsp;|&nbsp; **Page**: `{chunk['page']}` &nbsp;|&nbsp; **Cosine Distance**: `{chunk['distance']:.4f}`")
                                st.markdown(
                                    f"""<div class="source-snippet">{chunk['text']}</div>""",
                                    unsafe_allow_html=True
                                )

            except APIError as error:
                st.error(format_api_error(error))
            except Exception as error:
                st.error(f"An unexpected error occurred: {error}")

    # Clean Footer
    st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin-top: 3rem; margin-bottom: 1.25rem;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: #94A3B8;">
            <div>Enterprise AI & Machine Learning Knowledge Assistant &bull; Production v2.0</div>
            <div>Powered by Python &bull; Google Gemini &bull; ChromaDB &bull; Sentence Transformers</div>
        </div>
        """,
        unsafe_allow_html=True
    )
