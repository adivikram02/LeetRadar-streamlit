import streamlit as st
import joblib


@st.cache_resource(show_spinner=False)
def load_company_lookup():
    return joblib.load('./models/company_lookup.joblib')


@st.cache_resource(show_spinner=False)
def load_company_tfidf():
    return joblib.load('./models/company_tfidf.joblib')


@st.cache_resource(show_spinner=False)
def load_company_model():
    return joblib.load('./models/company_model.joblib')


@st.cache_resource(show_spinner=False)
def load_company_mlb():
    return joblib.load('./models/company_mlb.joblib')


company_lookup = load_company_lookup()
tfidf          = load_company_tfidf()
company_model  = load_company_model()
mlb            = load_company_mlb()


def find_companies(title='', description='', pattern='', difficulty='', threshold=0.30):
    """
    Find which companies asked a given LeetCode problem.
    - If the problem is in the lookup  -> return exact companies from data
    - If not                           -> use ML model to predict likely companies
    """
    if not title and not description:
        return {'error': 'Please provide at least a title or description'}

    title_lower = title.strip().lower()

    # Direct lookup first — most accurate
    if title_lower and title_lower in company_lookup:
        data = company_lookup[title_lower]
        return {
            'title':           title,
            'companies':       data['companies'],
            'num_occurrences': data['num_occurrences'],
            'difficulty':      data.get('difficulty', ''),
            'pattern':         data.get('pattern', ''),
            'source':          'lookup',
            'note':            f"Found in dataset. Asked by {len(data['companies'])} companies.",
        }

    # Fallback to ML model
    combined = f"{title} {title} {title} {pattern} {pattern} {difficulty} {description}"
    X_input  = tfidf.transform([combined])
    raw      = company_model.predict_proba(X_input)

    # normalize to flat probability array
    # OneVsRestClassifier can return either a 2D array or a list of arrays
    # depending on the sklearn version
    if isinstance(raw, list):
        # older sklearn: list of (n_samples, 2) arrays — take positive class prob
        probs = [p[0][1] for p in raw]
    else:
        # newer sklearn: (n_samples, n_classes) array
        probs = list(raw[0])

    # try with threshold first
    predicted = [
        company
        for company, prob in zip(mlb.classes_, probs)
        if prob >= threshold
    ]

    # if nothing passes threshold, return top 3 as low-confidence predictions
    # so the user always gets some output rather than nothing
    if not predicted:
        top3_idx  = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
        predicted = [mlb.classes_[i] for i in top3_idx]
        note      = 'Not found in dataset. Showing top 3 predicted companies (low confidence).'
    else:
        note = f'Not found in dataset. Predicted {len(predicted)} likely companies.'

    return {
        'title':     title,
        'companies': predicted,
        'source':    'predicted',
        'note':      note,
    }