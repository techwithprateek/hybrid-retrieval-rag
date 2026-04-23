# 🎫 RAG Support Ticket Classifier (Hybrid Search)

A beginner-friendly **Retrieval-Augmented Generation (RAG)** system that classifies
customer support complaints and recommends resolution steps using:

- **Hybrid search** — semantic (FAISS + OpenAI embeddings) + keyword (TF-IDF)
- **LLM generation** — GPT-4o-mini with structured prompting
- **Streamlit UI** — interactive web interface

---

## 🗂️ Project Structure

```
hybrid-retrieval-rag/
├── app.py                    # Streamlit application (entry point)
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment variable file
├── data/
│   └── knowledge_base.csv    # 30 support issue entries with resolution steps
└── src/
    ├── retrieval.py           # HybridRetriever (FAISS + TF-IDF)
    └── llm.py                 # LLM response generator (OpenAI)
```

---

## 🔄 How It Works

```
Customer Complaint (free text)
        │
        ▼
┌──────────────────────────────────────┐
│         Hybrid Retrieval             │
│  ┌───────────────┐  ┌─────────────┐  │
│  │ Semantic FAISS│  │TF-IDF Cosine│  │
│  │ (embeddings)  │  │ (keywords)  │  │
│  └───────┬───────┘  └──────┬──────┘  │
│          └────── α ────────┘         │
│            Weighted sum              │
└──────────────────┬───────────────────┘
                   │ Top-k context entries
                   ▼
        ┌─────────────────────┐
        │  OpenAI GPT-4o-mini │
        │  Structured prompt  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Structured Output  │
        │  - Category         │
        │  - Subcategory      │
        │  - Journey Stage    │
        │  - Confidence       │
        │  - Resolution Steps │
        └─────────────────────┘
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/techwithprateek/hybrid-retrieval-rag.git
cd hybrid-retrieval-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and replace "sk-your-api-key-here" with your actual key
```

Or export it directly in your shell:

```bash
export OPENAI_API_KEY="sk-..."   # macOS / Linux
set OPENAI_API_KEY=sk-...        # Windows
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧑‍💻 Usage

1. Enter your **OpenAI API key** in the sidebar (or set it in `.env`).
2. Adjust **top-k** (number of context entries) and **α** (semantic vs keyword weight).
3. Type a customer complaint or pick a sample from the dropdown.
4. Click **Classify & Resolve**.
5. View the structured classification and resolution steps.
6. Expand the **Retrieved Knowledge Base Context** section to see how hybrid scoring works.

---

## 🧠 Key Concepts

| Concept | Description |
|---|---|
| **RAG** | Retrieve relevant knowledge, then generate a grounded response |
| **Semantic search** | FAISS index over OpenAI embeddings — understands *meaning* |
| **Keyword search** | TF-IDF cosine similarity — captures *exact terms* like "OTP", "UPI" |
| **Hybrid scoring** | `score = α × semantic + (1−α) × keyword` |
| **Structured prompt** | Forces the LLM to output valid JSON with a fixed schema |

---

## 🧱 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit | Web UI |
| FAISS | Vector index for semantic search |
| scikit-learn | TF-IDF vectorizer |
| OpenAI API | Embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`) |
| pandas / numpy | Data handling |

---

## 📚 Knowledge Base

`data/knowledge_base.csv` contains **30 support ticket templates** across 8 categories:

| Category | Examples |
|---|---|
| Payment | Failed transaction, UPI failure, refund, OTP, duplicate charge |
| Account | Login failure, account locked, KYC, profile update |
| Order | Order not placed, wrong item, delayed, cancellation, return |
| Delivery | Damaged package, wrong address, agent issue |
| Technical | App crash, slow loading, search failure, gateway error |
| Coupon | Code not applied, cashback not credited |
| Subscription | Not activated, auto-renewal dispute |
| Wallet | Balance not updated, payment failed |
| Fraud | Unauthorized transaction, account hacked |

Each entry includes: `category`, `subcategory`, `journey_stage`, `issue_description`, and `resolution_steps`.

---

## 🔧 Extending the Project

- **Add more knowledge base entries** — edit `data/knowledge_base.csv`
- **Swap the LLM** — change `LLM_MODEL` in `src/llm.py`
- **Change the embedding model** — change `EMBEDDING_MODEL` in `src/retrieval.py`
- **Add reranking** — insert a cross-encoder reranker between retrieval and generation
- **Persist the FAISS index** — use `faiss.write_index` to avoid recomputing on every run
- **Add evaluation** — log retrieved entries and LLM outputs to measure RAG quality
