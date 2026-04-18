import joblib
import numpy as np
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_difficulty_model():
    return joblib.load('./models/difficulty_model.joblib')


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    return joblib.load('./models/difficulty_label_encoder.joblib')


@st.cache_resource(show_spinner=False)
def load_difficulty_lookup():
    return joblib.load('./models/difficulty_lookup.joblib')


@st.cache_resource(show_spinner=False)
def load_patterns_lookup():
    return joblib.load('./models/patterns_lookup.joblib')


difficulty_model  = load_difficulty_model()
label_enc         = load_label_encoder()
difficulty_lookup = load_difficulty_lookup()
patterns_lookup   = load_patterns_lookup()


def predict_difficulty(title: str, description: str = '', topics: str = '',
                       leetcode_label: str = None) -> dict:

    # use actual dataset patterns if title is known
    # avoids multi-pattern string from classifier interfering with prediction
    actual_topics = patterns_lookup.get(title.strip().lower(), topics)

    # Build combined text the same way as during training
    combined = f"{title} {title} {title} {actual_topics} {actual_topics} {description}"

    # Get probabilities — handle LinearSVC fallback
    clf = difficulty_model.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        probs = difficulty_model.predict_proba([combined])[0]
    else:
        scores = difficulty_model.decision_function([combined])[0]
        scores = np.exp(scores - scores.max())
        probs  = scores / scores.sum()

    # Decode class indices back to string labels
    class_ints = difficulty_model.classes_
    classes    = label_enc.inverse_transform(class_ints)

    predicted  = classes[np.argmax(probs)]
    confidence = round(float(np.max(probs)) * 100, 1)

    prob_breakdown = {
        cls: f'{round(float(p) * 100, 1)}%'
        for cls, p in zip(classes, probs)
    }

    # auto lookup actual label from saved dictionary if not explicitly provided
    if not leetcode_label:
        leetcode_label = difficulty_lookup.get(title.strip().lower(), None)

    actual_label = leetcode_label if leetcode_label else 'Not in dataset'

    mislabelled_flag = (
        bool(leetcode_label)
        and leetcode_label != predicted
        and confidence >= 70
    )

    return {
        'predicted_difficulty': predicted,
        'confidence':           f'{confidence}%',
        'probabilities':        prob_breakdown,
        'leetcode_label':       actual_label,
        'mislabelled_flag':     mislabelled_flag,
        'note': (
            f'Possibly mislabelled! LeetCode says {leetcode_label}, '
            f'model predicts {predicted} with {confidence}% confidence.'
            if mislabelled_flag else 'Label matches prediction.'
        )
    }