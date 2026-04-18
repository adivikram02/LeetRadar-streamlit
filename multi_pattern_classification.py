import joblib
from sklearn.preprocessing import normalize
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_lr_classifier():
    lr_classifier = joblib.load("./models/pattern_lr_v2.joblib")
    # patch missing attribute removed in newer sklearn versions
    for estimator in lr_classifier.estimators_:
        if not hasattr(estimator, 'multi_class'):
            estimator.multi_class = 'ovr'
    return lr_classifier

@st.cache_resource(show_spinner=False)
def load_mlb():
    mlb = joblib.load("./models/multilabel_binarizer_v2.joblib")
    return mlb

@st.cache_resource(show_spinner=False)
def load_per_label_thresholds_pr():
    per_label_thresholds_pr = joblib.load("./models/per_label_thresholds_v2.joblib")
    return per_label_thresholds_pr

@st.cache_resource(show_spinner=False)
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return embed_model

# lr_classifier = joblib.load("./models/pattern_lr_v2.joblib")
# mlb = joblib.load("./models/multilabel_binarizer_v2.joblib")
# per_label_thresholds_pr = joblib.load("./models/per_label_thresholds_v2.joblib")
# embed_model = joblib.load("./models/embed_model_v2.joblib")

lr_classifier = load_lr_classifier()
mlb = load_mlb()
per_label_thresholds_pr = load_per_label_thresholds_pr()
embed_model = load_embed_model()

@st.cache_data(show_spinner=False)
def predict_patterns(title: str, description: str = '') -> list:
    """
    Predict patterns for a LeetCode problem.

    Args:
        title       : problem title
        description : problem description (optional but improves accuracy)

    Returns:
        list of (pattern, confidence) tuples sorted by confidence descending
    """
    text  = (title + ' | ' + description).strip(' |')
    vec   = normalize(embed_model.encode([text], convert_to_numpy=True), norm='l2')
    proba = lr_classifier.predict_proba(vec)[0]

    results = [
        (mlb.classes_[i], round(float(proba[i]), 4))
        for i in range(len(proba))
        if proba[i] >= per_label_thresholds_pr[mlb.classes_[i]]
    ]

    if not results:
        top2_idx = np.argsort(proba)[::-1][:2]
        results  = [(mlb.classes_[i], round(float(proba[i]), 4)) for i in top2_idx]

    return sorted(results, key=lambda x: -x[1])