"""
Hybrid Retrieval System
Combines semantic search (FAISS + OpenAI embeddings) with keyword search (TF-IDF)
to retrieve the most relevant knowledge base entries for a given customer complaint.
"""

import os
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class HybridRetriever:
    """
    Retrieves relevant support knowledge base entries using a hybrid of:
    - Semantic search via OpenAI embeddings + FAISS
    - Keyword search via TF-IDF cosine similarity
    """

    def __init__(self, knowledge_base_path: str, alpha: float = 0.6):
        """
        Args:
            knowledge_base_path: Path to the CSV knowledge base file.
            alpha: Weight for semantic search (1 - alpha for keyword search).
                   Higher alpha means more weight on semantic similarity.
        """
        self.alpha = alpha
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        self.df = pd.read_csv(knowledge_base_path)
        self._build_search_text()
        self._build_tfidf_index()
        self._build_faiss_index()

    def _build_search_text(self) -> None:
        """Combine category, subcategory, and issue description into a single search string."""
        self.df["search_text"] = (
            self.df["category"].fillna("")
            + " "
            + self.df["subcategory"].fillna("")
            + " "
            + self.df["issue_description"].fillna("")
        )

    def _build_tfidf_index(self) -> None:
        """Fit a TF-IDF vectorizer on all knowledge base entries."""
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=10_000,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["search_text"])

    def _build_faiss_index(self) -> None:
        """Compute OpenAI embeddings for all entries and build a FAISS index."""
        texts = self.df["search_text"].tolist()
        embeddings = self._embed(texts)
        embeddings = normalize(embeddings, norm="l2")
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.faiss_index.add(embeddings.astype(np.float32))

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return a numpy array of embeddings for a list of texts."""
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def _semantic_scores(self, query: str) -> np.ndarray:
        """Return FAISS inner-product similarity scores (higher = more similar)."""
        query_vec = self._embed([query])
        query_vec = normalize(query_vec, norm="l2").astype(np.float32)
        n = len(self.df)
        scores, indices = self.faiss_index.search(query_vec, n)
        # Re-order to match original dataframe order
        ordered = np.zeros(n, dtype=np.float32)
        ordered[indices[0]] = scores[0]
        # Shift to [0, 1] — inner-product of L2-normalised vectors is in [-1, 1]
        ordered = (ordered + 1) / 2
        return ordered

    def _keyword_scores(self, query: str) -> np.ndarray:
        """Return TF-IDF cosine similarity scores for the query."""
        query_vec = self.tfidf.transform([query])
        # Cosine similarity = dot product of L2-normalised vectors
        tfidf_norm = normalize(self.tfidf_matrix, norm="l2")
        query_norm = normalize(query_vec, norm="l2")
        scores = (tfidf_norm @ query_norm.T).toarray().flatten()
        return scores.astype(np.float32)

    def retrieve(self, query: str, top_k: int = 3) -> pd.DataFrame:
        """
        Retrieve the top-k most relevant knowledge base entries.

        Args:
            query: The customer complaint text.
            top_k: Number of results to return.

        Returns:
            DataFrame with the top-k rows from the knowledge base, plus
            a 'hybrid_score' column showing the combined relevance score.
        """
        semantic = self._semantic_scores(query)
        keyword = self._keyword_scores(query)

        # Normalise each score vector to [0, 1] to make them comparable
        def _minmax(arr: np.ndarray) -> np.ndarray:
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

        hybrid = self.alpha * _minmax(semantic) + (1 - self.alpha) * _minmax(keyword)

        top_indices = np.argsort(hybrid)[::-1][:top_k]
        result = self.df.iloc[top_indices].copy()
        result["hybrid_score"] = hybrid[top_indices]
        result["semantic_score"] = semantic[top_indices]
        result["keyword_score"] = keyword[top_indices]
        return result.reset_index(drop=True)
