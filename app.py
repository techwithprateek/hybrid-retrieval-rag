"""
RAG Support Ticket Classifier — Streamlit App
Classifies customer support complaints using hybrid retrieval (FAISS + TF-IDF)
combined with an LLM to generate structured resolutions.
"""

import os
import sys
import streamlit as st
import streamlit as st

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.csv")

from src.retrieval import HybridRetriever
from src.llm import generate_response

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sample complaints for quick demos
# ---------------------------------------------------------------------------
SAMPLE_COMPLAINTS = [
    "Select a sample complaint…",
    "I paid via UPI but my order was not placed. Money got deducted from my account.",
    "I have been trying to log in but it says my account is locked. I entered the wrong password a few times.",
    "My package arrived damaged. The box was completely crushed and the product inside is broken.",
    "I never received my refund even though I returned the item 2 weeks ago.",
    "The app keeps crashing every time I try to open it on my phone.",
    "Someone made a purchase from my account without my knowledge. I didn't authorize this transaction.",
    "My coupon code is not working at checkout. It says invalid.",
    "I received a completely different product than what I ordered.",
]

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4ff;
        border-left: 4px solid #4c6ef5;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .confidence-high { color: #2e7d32; font-weight: 700; }
    .confidence-medium { color: #f57c00; font-weight: 700; }
    .confidence-low { color: #c62828; font-weight: 700; }
    .step-item {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state["openai_api_key"],
        help="Enter your OpenAI API key. It is used only for this session.",
        placeholder="sk-...",
    )
    if api_key != st.session_state["openai_api_key"]:
        st.session_state["openai_api_key"] = api_key

    st.divider()

    st.subheader("🔧 Retrieval Settings")
    top_k = st.slider(
        "Number of context entries (top-k)",
        min_value=1,
        max_value=5,
        value=3,
        help="How many knowledge base entries to retrieve for each query.",
    )
    alpha = st.slider(
        "Semantic weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="α = weight for semantic search; (1−α) = weight for keyword search.",
    )

    st.divider()
    st.caption("**How it works**")
    st.caption(
        "1. Your complaint is searched against the knowledge base using both "
        "semantic embeddings (FAISS) and keyword matching (TF-IDF).\n\n"
        "2. The top results are passed to an LLM with a structured prompt.\n\n"
        "3. The LLM returns a classification and resolution steps."
    )

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.markdown(
    '<p class="main-header">🎫 RAG Support Ticket Classifier</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Hybrid Search (FAISS + TF-IDF) &nbsp;|&nbsp; '
    'Retrieval-Augmented Generation &nbsp;|&nbsp; OpenAI GPT</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load retriever
# ---------------------------------------------------------------------------
def load_retriever(kb_path: str, _alpha: float) -> HybridRetriever:
    return HybridRetriever(kb_path, alpha=_alpha)


# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
st.subheader("📝 Customer Complaint")

sample = st.selectbox("Quick demo — pick a sample complaint:", SAMPLE_COMPLAINTS)
complaint_input = st.text_area(
    "Or type your complaint here:",
    value="" if sample == SAMPLE_COMPLAINTS[0] else sample,
    height=120,
    placeholder="e.g. I paid via UPI but my order was not placed…",
)

classify_btn = st.button("🔍 Classify & Resolve", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
if classify_btn:
    complaint = complaint_input.strip()

    if not complaint:
        st.warning("Please enter a complaint before classifying.")
        st.stop()

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    try:
        retriever = load_retriever(KNOWLEDGE_BASE_PATH, alpha)
    except Exception as exc:
        st.error(f"Failed to build the search index: {exc}")
        st.stop()

    with st.spinner("Retrieving relevant context…"):
        try:
            retrieved = retriever.retrieve(complaint, top_k=top_k)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()

    with st.spinner("Generating resolution with LLM…"):
        try:
            result = generate_response(complaint, retrieved)
        except Exception as exc:
            st.error(f"LLM generation failed: {exc}")
            st.stop()

    st.success("✅ Classification complete!")
    st.divider()

    # -----------------------------------------------------------------------
    # Results — classification cards
    # -----------------------------------------------------------------------
    st.subheader("🏷️ Classification")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Category</div>'
            f'<div class="metric-value">{result.get("category", "—")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Subcategory</div>'
            f'<div class="metric-value">{result.get("subcategory", "—")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Journey Stage</div>'
            f'<div class="metric-value">{result.get("journey_stage", "—")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col4:
        confidence = result.get("confidence", "—")
        css_class = {
            "High": "confidence-high",
            "Medium": "confidence-medium",
            "Low": "confidence-low",
        }.get(confidence, "")
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Confidence</div>'
            f'<div class="metric-value {css_class}">{confidence}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    if result.get("summary"):
        st.info(f"**Summary:** {result['summary']}")

    # -----------------------------------------------------------------------
    # Resolution steps
    # -----------------------------------------------------------------------
    st.subheader("🛠️ Resolution Steps")
    steps = result.get("resolution_steps", [])
    if steps:
        for i, step in enumerate(steps, 1):
            st.markdown(
                f'<div class="step-item"><strong>Step {i}.</strong> {step}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("No resolution steps were generated.")

    # -----------------------------------------------------------------------
    # Retrieved context (expandable)
    # -----------------------------------------------------------------------
    st.divider()
    with st.expander("🔎 Retrieved Knowledge Base Context", expanded=False):
        st.caption(
            f"Top {len(retrieved)} entries retrieved using hybrid search "
            f"(α={alpha:.2f} semantic, {1-alpha:.2f} keyword)"
        )
        for _, row in retrieved.iterrows():
            with st.container():
                cols = st.columns([3, 1, 1, 1])
                cols[0].markdown(f"**{row['category']} › {row['subcategory']}**")
                cols[1].metric("Hybrid", f"{row['hybrid_score']:.3f}")
                cols[2].metric("Semantic", f"{row['semantic_score']:.3f}")
                cols[3].metric("Keyword", f"{row['keyword_score']:.3f}")
                st.markdown(
                    f"_{row['issue_description'][:180]}…_"
                    if len(row["issue_description"]) > 180
                    else f"_{row['issue_description']}_"
                )
                st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<br><hr><center><small>Built with Streamlit · FAISS · scikit-learn · OpenAI</small></center>",
    unsafe_allow_html=True,
)
