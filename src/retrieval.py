"""
Hybrid Retrieval System
Combines semantic search (FAISS + OpenAI embeddings) with keyword search (TF-IDF)
to retrieve the most relevant knowledge base entries for a given customer complaint.

Why hybrid retrieval?
- Semantic search understands *meaning* (e.g. "money deducted" ≈ "payment failed")
  but can miss exact technical terms like "UPI", "OTP", or error codes.
- Keyword (TF-IDF) search captures *exact terms* well but doesn't understand
  paraphrasing or synonyms.
- Combining both gives better recall and precision than either method alone.
"""

import os
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from openai import OpenAI

# The OpenAI embedding model used to convert text → dense vectors.
# "text-embedding-3-small" is the smallest, cheapest model and works well for
# short support-ticket texts. Its output dimension is always 1536.
EMBEDDING_MODEL = "text-embedding-3-small"

# The vector dimension produced by text-embedding-3-small.
# Used to initialise the FAISS index — must match the actual embedding size.
EMBEDDING_DIM = 1536


class HybridRetriever:
    """
    Retrieves relevant support knowledge base entries using a hybrid of:
    - Semantic search via OpenAI embeddings + FAISS (understands meaning)
    - Keyword search via TF-IDF cosine similarity (captures exact terms)

    The final score blends the two signals:
        hybrid_score = alpha * semantic_score + (1 - alpha) * keyword_score
    """

    def __init__(self, knowledge_base_path: str, alpha: float = 0.6):
        """
        Load the knowledge base from CSV, build both search indexes, and
        set up the OpenAI client for generating embeddings.

        Args:
            knowledge_base_path: Path to the CSV knowledge base file.
            alpha: Weight for semantic search (0.0 – 1.0).
                   - alpha=1.0 → pure semantic search
                   - alpha=0.0 → pure keyword search
                   - alpha=0.6 → 60% semantic, 40% keyword (good default)
        """
        # Store alpha so _retrieve() can use it later
        self.alpha = alpha

        # Read the API key from the environment (set via .env or the sidebar).
        # We raise a clear error instead of a cryptic KeyError if it's missing.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before creating HybridRetriever."
            )
        # Create the OpenAI client — this is used only for embedding generation
        self.client = OpenAI(api_key=api_key)

        # Load the knowledge base CSV into a pandas DataFrame so we can
        # easily slice, filter, and attach score columns later
        self.df = pd.read_csv(knowledge_base_path)

        # Build a combined text field and both search indexes at startup.
        # Doing this once (not per query) means queries are fast.
        self._build_search_text()
        self._build_tfidf_index()
        self._build_faiss_index()

    def _build_search_text(self) -> None:
        """
        Create a single 'search_text' column by joining the category,
        subcategory, and issue description for every row.

        Why combine all three fields?
        - The category and subcategory act as strong keyword signals
          (e.g. "Payment UPI Failure").
        - The issue description adds semantic context.
        - Using one combined string keeps both indexes simple — they each
          operate on a single field per row.
        """
        self.df["search_text"] = (
            self.df["category"].fillna("")
            + " "
            + self.df["subcategory"].fillna("")
            + " "
            + self.df["issue_description"].fillna("")
        )

    def _build_tfidf_index(self) -> None:
        """
        Fit a TF-IDF vectorizer on every row's search_text and store the
        resulting sparse matrix for fast dot-product scoring at query time.

        TF-IDF (Term Frequency – Inverse Document Frequency) scores each word
        by how often it appears in a document vs. how common it is across all
        documents.  Rare, domain-specific words like "OTP" or "UPI" get high
        scores; common filler words get near-zero scores.

        Configuration choices:
        - ngram_range=(1, 2): Include both single words ("UPI") and bigrams
          ("UPI failure") to capture phrase-level matches.
        - stop_words="english": Ignore words like "the", "is", "a" that carry
          no meaning and would dilute similarity scores.
        - max_features=10_000: Cap the vocabulary at 10,000 terms to keep
          the matrix small and computations fast.
        """
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),      # capture single words AND two-word phrases
            stop_words="english",    # remove common English filler words
            max_features=10_000,     # vocabulary cap for efficiency
        )
        # fit_transform: (1) learns the vocabulary from all rows, then
        # (2) converts each row into a sparse TF-IDF vector.
        # Shape: (num_rows, vocab_size) as a sparse matrix.
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["search_text"])

        # Pre-compute the L2-normalised version of the TF-IDF matrix once.
        # This means _keyword_scores() can compute cosine similarity with a
        # simple dot product, without re-normalising the whole matrix every time.
        self.tfidf_matrix_norm = normalize(self.tfidf_matrix, norm="l2")

    def _build_faiss_index(self) -> None:
        """
        Embed every knowledge base entry using OpenAI and store the vectors
        in a FAISS index for fast nearest-neighbour search.

        Why FAISS?
        - Facebook AI Similarity Search (FAISS) is purpose-built for searching
          millions of dense vectors in milliseconds.
        - IndexFlatIP performs exact inner-product (dot product) search — when
          vectors are L2-normalised, inner product equals cosine similarity.

        Why L2-normalise before adding to the index?
        - Cosine similarity = dot(a, b) / (||a|| * ||b||).
        - If both vectors are already unit-length (L2 norm = 1), the
          denominator is always 1, so cosine similarity = simple dot product.
        - This lets us use the fast IndexFlatIP index for cosine similarity.
        """
        texts = self.df["search_text"].tolist()

        # Call OpenAI to convert all knowledge base texts into dense vectors.
        # This is the only network call in the constructor — subsequent queries
        # only embed the (short) query string, not the whole KB.
        embeddings = self._embed(texts)

        # L2-normalise so that inner product == cosine similarity in the index
        embeddings = normalize(embeddings, norm="l2")

        # Create an exact inner-product FAISS index with the correct dimension
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)

        # Add all knowledge base vectors to the index.
        # FAISS requires float32 inputs.
        self.faiss_index.add(embeddings.astype(np.float32))

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Call the OpenAI Embeddings API and return results as a numpy array.

        Args:
            texts: A list of strings to embed (can be one or many).

        Returns:
            A (len(texts), EMBEDDING_DIM) float32 array where each row is
            the dense embedding of the corresponding input string.
        """
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        # response.data is a list of Embedding objects; extract the raw vectors
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def _semantic_scores(self, query: str) -> np.ndarray:
        """
        Embed the query and compute cosine similarity against every KB entry
        via the FAISS index.

        Returns a 1-D array of length (num_rows,) where higher = more similar.

        Why shift the scores from [-1, 1] to [0, 1]?
        - Inner product of L2-normalised vectors (= cosine similarity) can be
          negative for very dissimilar texts.
        - Shifting to [0, 1] makes the semantic scores compatible with the
          keyword scores (which are inherently ≥ 0) for the hybrid blend.
        """
        # Embed and normalise the query the same way we normalised the KB vectors
        query_vec = self._embed([query])
        query_vec = normalize(query_vec, norm="l2").astype(np.float32)

        n = len(self.df)

        # FAISS search returns scores and their *original* indices, sorted by
        # descending score. We ask for ALL n entries so no result is missed.
        scores, indices = self.faiss_index.search(query_vec, n)

        # FAISS returns results in score order, not in KB order.
        # Re-map them back to match the original DataFrame row positions so
        # we can later align semantic scores with keyword scores by index.
        ordered = np.zeros(n, dtype=np.float32)
        ordered[indices[0]] = scores[0]

        # Shift cosine similarity from [-1, 1] → [0, 1] for easier blending.
        # Entries with score 1.0 are maximally similar; 0.5 are neutral; 0.0 opposite.
        ordered = (ordered + 1) / 2
        return ordered

    def _keyword_scores(self, query: str) -> np.ndarray:
        """
        Transform the query with the fitted TF-IDF vectorizer, then compute
        cosine similarity against every KB entry using the pre-normalised matrix.

        Returns a 1-D float32 array of length (num_rows,) with values in [0, 1].

        Why cosine similarity for keyword search?
        - Cosine similarity measures the angle between two vectors, ignoring
          document length.  A long KB entry and a short complaint can still
          score 1.0 if they share the same important words.
        - Using the pre-normalised KB matrix (self.tfidf_matrix_norm) avoids
          re-normalising the entire matrix on every query call.
        """
        # Transform the query into a TF-IDF sparse vector using the already-
        # fitted vocabulary (same feature space as the KB matrix)
        query_vec = self.tfidf.transform([query])

        # L2-normalise the query vector so the dot product equals cosine similarity
        query_norm = normalize(query_vec, norm="l2")

        # Matrix multiply: (num_rows × vocab) · (vocab × 1) → (num_rows × 1)
        # .toarray().flatten() converts the sparse result to a plain 1-D array
        scores = (self.tfidf_matrix_norm @ query_norm.T).toarray().flatten()
        return scores.astype(np.float32)

    def retrieve(self, query: str, top_k: int = 3) -> pd.DataFrame:
        """
        Retrieve the top-k most relevant knowledge base entries for a query
        using the blended hybrid score.

        Algorithm:
        1. Compute raw semantic scores (cosine similarity via FAISS).
        2. Compute raw keyword scores (TF-IDF cosine similarity).
        3. Min-max normalise each score vector independently to [0, 1] so
           the two signals are on the same scale before blending.
        4. Blend:  hybrid = alpha * semantic_norm + (1 - alpha) * keyword_norm
        5. Return the top-k rows sorted by hybrid score, descending.

        Args:
            query: The customer complaint text.
            top_k: Number of results to return (typically 3–5).

        Returns:
            DataFrame with the top-k KB rows plus three extra columns:
            - hybrid_score   — the blended relevance score used for ranking
            - semantic_score — raw (shifted) cosine similarity from embeddings
            - keyword_score  — raw TF-IDF cosine similarity
        """
        # Step 1 & 2: Get raw scores from both retrieval methods
        semantic = self._semantic_scores(query)
        keyword = self._keyword_scores(query)

        # Step 3: Min-max normalisation
        # Why? semantic scores live in [0, 1] (after shifting) but their
        # actual range per query may be, say, [0.45, 0.72].  Keyword scores
        # may live in [0, 0.30].  Without normalisation, the semantic signal
        # would dominate simply because of its larger absolute values.
        # After normalisation, both signals span the full [0, 1] range and
        # alpha truly controls the trade-off between them.
        def _minmax(arr: np.ndarray) -> np.ndarray:
            rng = arr.max() - arr.min()
            # Guard against a flat array (all scores identical) → return zeros
            # to avoid division by zero; no entry is preferred over another.
            return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

        # Step 4: Blend the two normalised score vectors
        hybrid = self.alpha * _minmax(semantic) + (1 - self.alpha) * _minmax(keyword)

        # Step 5: Pick the top-k indices in descending hybrid score order
        # argsort gives ascending order; [::-1] reverses to descending; [:top_k] takes first k
        top_indices = np.argsort(hybrid)[::-1][:top_k]

        # Build the result DataFrame from the selected rows
        result = self.df.iloc[top_indices].copy()

        # Attach the three score columns for transparency (shown in the UI expander)
        result["hybrid_score"] = hybrid[top_indices]
        result["semantic_score"] = semantic[top_indices]
        result["keyword_score"] = keyword[top_indices]

        # Reset index so the caller always gets 0-based row numbers
        return result.reset_index(drop=True)
