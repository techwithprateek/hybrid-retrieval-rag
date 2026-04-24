"""
RAG Support Ticket Classifier — Streamlit App
Classifies customer support complaints using hybrid retrieval (FAISS + TF-IDF)
combined with an LLM to generate structured resolutions.

Architecture (high level):
1. User types (or picks) a complaint in the UI.
2. HybridRetriever searches the knowledge base using both semantic embeddings
   (FAISS) and keyword matching (TF-IDF) and returns the top-k most relevant entries.
3. The top-k entries are passed as context to generate_response(), which sends
   the complaint + context to GPT-4o-mini and gets back a structured JSON response.
4. The structured response (category, subcategory, confidence, steps…) is displayed
   in the Streamlit UI as classification cards and a numbered resolution list.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Startup: resolve file paths and load environment variables
# ---------------------------------------------------------------------------
# BASE_DIR = the directory where this script lives (the project root).
# We compute it once here so that relative paths (like data/knowledge_base.csv)
# work correctly regardless of where the user runs `streamlit run app.py` from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from a .env file in the project root (if it exists).
# This lets users put OPENAI_API_KEY=sk-... in .env instead of exporting it manually.
# load_dotenv() is a no-op if the file doesn't exist, so this is always safe to call.
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Absolute path to the knowledge base CSV used by HybridRetriever
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "data", "knowledge_base.csv")

# ---------------------------------------------------------------------------
# Local imports — must come after load_dotenv so the key is available
# ---------------------------------------------------------------------------
from src.retrieval import HybridRetriever   # noqa: E402  (import after path setup)
from src.llm import generate_response        # noqa: E402

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
# st.set_page_config() MUST be the very first Streamlit call in the script.
# It sets the browser tab title, favicon, and default layout.
st.set_page_config(
    page_title="RAG Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",              # use the full browser width (better for the 4-column cards)
    initial_sidebar_state="expanded",  # show the sidebar open by default
)

# ---------------------------------------------------------------------------
# Sample complaints for quick demos
# ---------------------------------------------------------------------------
# These are pre-written complaints covering all nine KB categories.
# Selecting one auto-fills the text area so users can demo the classifier
# without typing anything themselves.
SAMPLE_COMPLAINTS = [
    "Select a sample complaint…",   # placeholder / default (no-op) option
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
# Custom CSS — injected once at startup
# ---------------------------------------------------------------------------
# Streamlit's default styling is functional but plain.  We inject a small
# CSS block to style the classification metric cards and resolution step items
# without needing an external CSS file.
st.markdown(
    """
    <style>
    /* Large bold page title */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    /* Smaller subtitle below the title */
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    /* Card container for each classification metric (category, subcategory, etc.) */
    .metric-card {
        background: #f0f4ff;
        border-left: 4px solid #4c6ef5;  /* accent stripe on the left */
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    /* Small all-caps label above each metric value */
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* The value itself (e.g. "Payment", "UPI Failure") */
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    /* Confidence colour coding: green / orange / red */
    .confidence-high   { color: #2e7d32; font-weight: 700; }
    .confidence-medium { color: #f57c00; font-weight: 700; }
    .confidence-low    { color: #c62828; font-weight: 700; }
    /* Container for each numbered resolution step */
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
# Sidebar — API key input and retrieval settings
# ---------------------------------------------------------------------------
# Everything inside `with st.sidebar:` renders in the collapsible left panel.
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- API key ---
    # We store the key in st.session_state (browser-tab-scoped) rather than
    # in os.environ directly here.  This prevents the key from leaking to other
    # users in a multi-user Streamlit deployment (each session has its own state).
    # The key is only written to os.environ immediately before each classification
    # call (see the `if classify_btn:` block below).
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""  # initialise to empty on first load

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",          # renders as dots so the key isn't visible on screen
        value=st.session_state["openai_api_key"],
        help="Enter your OpenAI API key. It is used only for this session.",
        placeholder="sk-...",
    )
    # Sync the text input value back to session_state whenever it changes.
    # Streamlit reruns the script on every interaction, so this keeps session_state
    # consistent with what's currently in the text box.
    if api_key != st.session_state["openai_api_key"]:
        st.session_state["openai_api_key"] = api_key

    st.divider()

    # --- Retrieval settings ---
    st.subheader("🔧 Retrieval Settings")

    # top_k: how many knowledge base entries to retrieve per query.
    # More entries = more context for the LLM, but also more tokens (cost) and
    # potential noise from less-relevant entries.  3 is a good default.
    top_k = st.slider(
        "Number of context entries (top-k)",
        min_value=1,
        max_value=5,
        value=3,
        help="How many knowledge base entries to retrieve for each query.",
    )

    # alpha (α): the blending weight between semantic and keyword search.
    # alpha=1.0 → 100% semantic (embedding cosine similarity)
    # alpha=0.0 → 100% keyword (TF-IDF cosine similarity)
    # alpha=0.6 → 60% semantic + 40% keyword (good general-purpose default)
    alpha = st.slider(
        "Semantic weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="α = weight for semantic search; (1−α) = weight for keyword search.",
    )

    st.divider()
    # Brief explanation of the pipeline for users who want to understand what's happening
    st.caption("**How it works**")
    st.caption(
        "1. Your complaint is searched against the knowledge base using both "
        "semantic embeddings (FAISS) and keyword matching (TF-IDF).\n\n"
        "2. The top results are passed to an LLM with a structured prompt.\n\n"
        "3. The LLM returns a classification and resolution steps."
    )

# ---------------------------------------------------------------------------
# Page title and subtitle
# ---------------------------------------------------------------------------
# We use raw HTML via st.markdown() instead of st.title()/st.subheader() to
# apply the custom CSS classes defined above (.main-header, .sub-header).
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
# Retriever factory function
# ---------------------------------------------------------------------------
def load_retriever(kb_path: str, _alpha: float) -> HybridRetriever:
    """
    Create and return a HybridRetriever instance for the given KB and alpha.

    This is a plain function (not cached) because HybridRetriever reads the
    API key from os.environ at construction time.  Caching it with
    @st.cache_resource would freeze the key used at first build — if the user
    updates the key in the sidebar the cached retriever would keep using the
    old key.  Building it fresh on each classification call ensures the current
    key is always used.

    Args:
        kb_path: Absolute path to the knowledge base CSV.
        _alpha:  Hybrid scoring weight (passed to HybridRetriever).

    Returns:
        A ready-to-use HybridRetriever with indexes built from the KB.
    """
    return HybridRetriever(kb_path, alpha=_alpha)


# ---------------------------------------------------------------------------
# Complaint input section
# ---------------------------------------------------------------------------
st.subheader("📝 Customer Complaint")

# Dropdown to pick a pre-written sample complaint for a quick demo.
# The first option is a placeholder — selecting it leaves the text area empty.
sample = st.selectbox("Quick demo — pick a sample complaint:", SAMPLE_COMPLAINTS)

# Main text area where the complaint lives.
# If a sample was picked (not the placeholder), pre-fill the text area with it;
# otherwise start with an empty string.
complaint_input = st.text_area(
    "Or type your complaint here:",
    value="" if sample == SAMPLE_COMPLAINTS[0] else sample,
    height=120,
    placeholder="e.g. I paid via UPI but my order was not placed…",
)

# Primary action button — triggers the full RAG pipeline when clicked
classify_btn = st.button("🔍 Classify & Resolve", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Classification pipeline — runs only when the button is clicked
# ---------------------------------------------------------------------------
if classify_btn:
    # Strip whitespace so a complaint that's all spaces is treated as empty
    complaint = complaint_input.strip()

    # Guard: refuse to run if no complaint text was provided
    if not complaint:
        st.warning("Please enter a complaint before classifying.")
        st.stop()  # halt the rest of the script for this rerun

    # Resolve the API key: check the sidebar session state first, then fall back
    # to the environment (which may have been populated from .env at startup).
    # This way both .env users and sidebar users are handled transparently.
    api_key = st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    # Write the resolved key to os.environ so that both HybridRetriever (which
    # calls os.getenv inside __init__) and generate_response pick up the right key.
    os.environ["OPENAI_API_KEY"] = api_key

    # Step 1: Build the retrieval index
    # This embeds the entire KB and builds FAISS + TF-IDF indexes.
    # It happens on every click (not cached) to always use the current API key.
    try:
        retriever = load_retriever(KNOWLEDGE_BASE_PATH, alpha)
    except Exception as exc:
        st.error(f"Failed to build the search index: {exc}")
        st.stop()

    # Step 2: Retrieve the top-k most relevant KB entries for the complaint
    with st.spinner("Retrieving relevant context…"):
        try:
            retrieved = retriever.retrieve(complaint, top_k=top_k)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()

    # Step 3: Send the complaint + retrieved context to the LLM for classification
    with st.spinner("Generating resolution with LLM…"):
        try:
            result = generate_response(complaint, retrieved)
        except Exception as exc:
            st.error(f"LLM generation failed: {exc}")
            st.stop()

    # Success banner — shown at the top so the user knows the pipeline finished
    st.success("✅ Classification complete!")
    st.divider()

    # -----------------------------------------------------------------------
    # Display: Classification metric cards
    # -----------------------------------------------------------------------
    # Four equal columns for: Category | Subcategory | Journey Stage | Confidence
    st.subheader("🏷️ Classification")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Custom HTML card (uses .metric-card CSS class defined above)
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
        # Confidence gets a colour based on its value (green/orange/red).
        # We look up the CSS class from the dict and embed it in the HTML.
        confidence = result.get("confidence", "—")
        css_class = {
            "High":   "confidence-high",    # green
            "Medium": "confidence-medium",   # orange
            "Low":    "confidence-low",      # red
        }.get(confidence, "")
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Confidence</div>'
            f'<div class="metric-value {css_class}">{confidence}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # One-sentence issue summary from the LLM, shown as an info banner
    if result.get("summary"):
        st.info(f"**Summary:** {result['summary']}")

    # -----------------------------------------------------------------------
    # Display: Resolution steps
    # -----------------------------------------------------------------------
    st.subheader("🛠️ Resolution Steps")
    steps = result.get("resolution_steps", [])
    if steps:
        # Render each step as a styled card with a bold "Step N." prefix.
        # We enumerate starting at 1 so step numbers are human-friendly (1, 2, 3…).
        for i, step in enumerate(steps, 1):
            st.markdown(
                f'<div class="step-item"><strong>Step {i}.</strong> {step}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("No resolution steps were generated.")

    # -----------------------------------------------------------------------
    # Display: Retrieved KB context (collapsible expander)
    # -----------------------------------------------------------------------
    # This section is hidden by default (expanded=False) so it doesn't clutter
    # the main view, but is available for users who want to understand *why*
    # the classifier produced a particular result.
    st.divider()
    with st.expander("🔎 Retrieved Knowledge Base Context", expanded=False):
        st.caption(
            f"Top {len(retrieved)} entries retrieved using hybrid search "
            f"(α={alpha:.2f} semantic, {1-alpha:.2f} keyword)"
        )
        # Show each retrieved entry with its three relevance scores side by side
        for _, row in retrieved.iterrows():
            with st.container():
                # 4 columns: KB entry title | hybrid score | semantic score | keyword score
                cols = st.columns([3, 1, 1, 1])
                cols[0].markdown(f"**{row['category']} › {row['subcategory']}**")
                cols[1].metric("Hybrid",   f"{row['hybrid_score']:.3f}")
                cols[2].metric("Semantic", f"{row['semantic_score']:.3f}")
                cols[3].metric("Keyword",  f"{row['keyword_score']:.3f}")
                # Show a truncated preview of the issue description (max 180 chars)
                # to keep the expander compact
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
