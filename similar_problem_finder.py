import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME       = 'all-MiniLM-L6-v2'
PATTERN_BONUS    = 0.15
MIN_PATTERN_CONF = 0.40


# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(MODEL_NAME, device='cpu')


@st.cache_resource(show_spinner=False)
def load_embeddings_index():
    return joblib.load('./models/similarity_embeddings_index.joblib')


@st.cache_resource(show_spinner=False)
def load_problem_df():
    return joblib.load('./models/similarity_problem_index.joblib')


@st.cache_resource(show_spinner=False)
def load_svm():
    return joblib.load('./models/pattern_svm.joblib')


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    return joblib.load('./models/label_encoder.joblib')


embed_model      = load_embed_model()
embeddings_index = load_embeddings_index()
problem_df       = load_problem_df()
svm              = load_svm()
le               = load_label_encoder()


# ── Core Function ──────────────────────────────────────────────────────────────
def predict_similar(
    title            : str,
    description      : str = '',
    topics           : str = '',
    top_k            : int = 10,
    difficulty_filter: str = None,
) -> list:
    """
    Find the top-k most similar LeetCode problems.

    Args:
        title:            Problem title
        description:      Problem description (optional)
        topics:           Related topics / patterns (optional)
        top_k:            Number of results to return (default 10)
        difficulty_filter: Filter results to 'Easy', 'Medium', or 'Hard' (optional)

    Returns:
        List of dicts with title, pattern, difficulty, similarity_score
    """
    # embed query the same way as during index build
    query_text = ' | '.join(v for v in [title, description, topics] if v)
    query_vec  = normalize(
        embed_model.encode([query_text], convert_to_numpy=True), norm='l2'
    )

    # cosine similarity against full index
    sim_scores = (embeddings_index @ query_vec.T).flatten()

    # pattern bonus
    svm_probs     = svm.predict_proba(query_vec)[0]
    query_pattern = le.classes_[np.argmax(svm_probs)]
    conf          = float(np.max(svm_probs))

    if conf >= MIN_PATTERN_CONF:
        mask       = (problem_df['pattern'].values == query_pattern).astype(float)
        sim_scores = sim_scores + PATTERN_BONUS * mask

    # exclude the query problem itself
    self_mask             = problem_df['title'].str.strip().str.lower() == title.strip().lower()
    sim_scores[self_mask] = -1.0

    # optional difficulty filter
    if difficulty_filter:
        allowed               = difficulty_filter.strip().capitalize()
        diff_mask             = problem_df['difficulty'].str.strip().str.capitalize() != allowed
        sim_scores[diff_mask] = -1.0

    top_idx = np.argsort(sim_scores)[::-1][:top_k]
    rows    = problem_df.iloc[top_idx]

    return [
        {
            'title'           : row['title'],
            'pattern'         : row['pattern'],
            'difficulty'      : row['difficulty'],
            'similarity_score': round(float(sim_scores[i]), 4),
        }
        for i, (_, row) in zip(top_idx, rows.iterrows())
    ]
