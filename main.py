import streamlit as st
import requests
from bs4 import BeautifulSoup
from multi_pattern_classification import predict_patterns
from pattern_explanation import get_pattern_explanation
from company_matching import find_companies
from difficulty_prediction import predict_difficulty
from similar_problem_finder import predict_similar

st.set_page_config(
    page_title="LeetRadar",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
* { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3rem; max-width: 900px; }
.stTextInput > div > div > input { border-radius: 8px; font-family: 'Inter', sans-serif; }
.stTextArea > div > div > textarea { border-radius: 8px; font-family: 'Inter', sans-serif; }
.stButton > button[kind="primary"] { background-color: #f5c542 !important; color: #1a1200 !important; border: none !important; font-weight: 700 !important; }
.stButton > button[kind="primary"]:hover { background-color: #e6b830 !important; color: #1a1200 !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { color: #ffffff !important; border-bottom-color: #f5c542 !important; }
.stTabs [data-baseweb="tab-list"] button { color: #888 !important; }
[data-testid="stToggle"] input:checked + div { background-color: #f5c542 !important; }
[data-testid="stToggle"] input:checked ~ div { background-color: #f5c542 !important; }
div[data-baseweb="toggle"] > div[aria-checked="true"] { background-color: #f5c542 !important; }
input[type="text"]:focus, textarea:focus { border-color: #f5c542 !important; box-shadow: 0 0 0 1px #f5c542 !important; outline: none !important; }
[data-baseweb="input"]:focus-within { border-color: #f5c542 !important; }
[data-baseweb="textarea"]:focus-within { border-color: #f5c542 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 2rem; padding-top: 1rem;">
  <div style="display: flex; align-items: center; gap: 14px;">
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="width: 48px; height: 48px; flex-shrink: 0;">
      <rect width="100" height="100" rx="22" fill="#f5c542"/>
      <text x="50" y="66" text-anchor="middle" font-family="'Courier New', Courier, monospace" font-size="36" font-weight="700" fill="#1a1200" letter-spacing="-1">&lt;/&gt;</text>
    </svg>
    <div>
      <p style="font-family: Inter, sans-serif; font-size: 28px; font-weight: 800; color: #f5c542; margin: 0; letter-spacing: -0.5px; line-height: 1;">LeetRadar</p>
      <p style="font-family: Inter, sans-serif; font-size: 13px; color: #888; margin: 5px 0 0;">A Coding Problem Pattern Recognition Tool</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if "title" not in st.session_state:
    st.session_state.title = None
if "description" not in st.session_state:
    st.session_state.description = None
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "char_count" not in st.session_state:
    st.session_state.char_count = 0


def string_to_slug(string: str):
    string_list = string.lower().split(" ")
    slug = "-".join(string_list)
    return slug


def patterns_to_string(patterns: list):
    length = len(patterns)
    if length == 0:
        return ""
    if length == 1:
        return str(patterns[0][0])
    if length == 2:
        return f"{patterns[0][0]} and {patterns[1][0]}"
    prefix = ", ".join([p[0] for p in patterns[:-1]])
    return f"{prefix} and {patterns[-1][0]}"


@st.cache_data
def fetch_from_leetcode(slug: str):
    try:
        url = "https://leetcode.com/graphql"
        headers = {"Content-Type": "application/json"}

        query = """
        query getQuestion($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                title
                content
            }
        }
        """

        resp = requests.post(
            url,
            json={"query": query, "variables": {"titleSlug": slug}},
            headers=headers,
            timeout=10
        )

        data = resp.json()["data"]["question"]

        if data is None:
            return None, None

        title   = data["title"]
        content = data["content"]

        if not content:
            return title, ""

        description = BeautifulSoup(content, "html.parser") \
                        .get_text(separator=" ") \
                        .strip()

        return title, description

    except Exception as e:
        print(f"Failed for {slug}: {e}")
        return None, None


# ── Input Section ──────────────────────────────────────────────────────────────

import_toggle = st.toggle("Import from LeetCode")

if import_toggle:
    with st.form("leetcode_form"):
        string = st.text_input(
            "Problem name",
            placeholder="e.g. two-sum or Two Sum"
        )
        slug = string_to_slug(string)
        import_button = st.form_submit_button("Import")
        if import_button:
            title, description = fetch_from_leetcode(slug)
            if title is None:
                st.error("Problem not found. Check the name and try again.")
            else:
                st.session_state.title = title
                st.session_state.description = description
                st.success(f"Fetched: {title}")
else:
    title = st.text_input(
        "Problem title",
        placeholder="e.g. Two Sum"
    )
    description = st.text_area(
        "Problem description",
        placeholder="Paste the full problem statement here...",
        height=180,
        key="desc_input"
    )
    char_count = len(description) if description else 0
    st.caption(f"{char_count} characters")
    st.session_state.title = title
    st.session_state.description = description


# ── Analyze Button ─────────────────────────────────────────────────────────────

st.markdown("")
col_analyze, col_clear = st.columns([1, 5])
with col_analyze:
    analyze_clicked = st.button("Analyze", type="primary")
with col_clear:
    if st.button("Clear", type="secondary"):
        st.session_state.title = None
        st.session_state.description = None
        st.session_state.show_results = False
        st.rerun()


# ── Results ────────────────────────────────────────────────────────────────────

if analyze_clicked and st.session_state.title and st.session_state.description:

    title       = st.session_state.title
    description = st.session_state.description

    with st.spinner("Analyzing patterns..."):
        patterns = predict_patterns(title, description)
    patterns_string = patterns_to_string(patterns)

    st.markdown("---")
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <p style="font-family: Inter, sans-serif; font-size: 22px; font-weight: 800; color: #f5c542; margin: 0;">Results</p>
            <p style="font-family: Inter, sans-serif; font-size: 13px; color: #888; margin: 4px 0 0;">Analysis for: <span style="color: #fff; font-weight: 600;">{}</span></p>
        </div>
    """.format(title), unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Patterns", "Explanation", "Difficulty", "Companies", "Similar Problems"
    ])

    # ── Patterns ──
    with tab1:
        st.markdown("""<p style="font-family:Inter,sans-serif;font-size:13px;font-weight:700;color:#f5c542;margin:0 0 14px;text-transform:uppercase;letter-spacing:0.05em;">Patterns</p>""", unsafe_allow_html=True)
        if patterns_string:
            def get_confidence_color(conf):
                if conf >= 75:
                    return "#22c55e"
                elif conf >= 50:
                    return "#f5c542"
                else:
                    return "#ef4444"

            for p in patterns:
                name  = p[0]
                conf  = round(p[1] * 100)
                color = get_confidence_color(conf)
                st.markdown(f"""<div style="background:#1e1e1e;border:0.5px solid #333;border-left:3px solid {color};border-radius:8px;padding:14px 18px;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;">
                <span style="font-family:Inter,sans-serif;font-size:15px;font-weight:600;color:#fff;">{name}</span>
                <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:100px;background:#333;border-radius:999px;height:6px;">
                <div style="width:{conf}%;background:{color};border-radius:999px;height:6px;"></div>
                </div>
                <span style="font-family:Inter,sans-serif;font-size:13px;font-weight:700;color:{color};min-width:36px;">{conf}%</span>
                </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No patterns detected.")

    # ── Explanation ──
    with tab2:
        st.markdown("""<p style="font-family:Inter,sans-serif;font-size:13px;font-weight:700;color:#f5c542;margin:0 0 14px;text-transform:uppercase;letter-spacing:0.05em;">Explanation</p>""", unsafe_allow_html=True)
        try:
            with st.spinner("Generating explanation..."):
                result = get_pattern_explanation(
                    title=title.lower(),
                    description=description,
                    pattern=patterns_string
                )
            explanation = result.get('pattern_explanation', '')
            if explanation:
                st.markdown(f"""<div style="background:#1e1e1e;border:0.5px solid #333;border-left:3px solid #f5c542;border-radius:8px;padding:20px 22px;margin-top:8px;">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
                <span style="font-family:Inter,sans-serif;font-size:12px;font-weight:700;background:#f5c54220;color:#f5c542;padding:3px 10px;border-radius:999px;">Pattern</span>
                <span style="font-family:Inter,sans-serif;font-size:13px;font-weight:600;color:#fff;">{patterns_string}</span>
                </div>
                <p style="font-family:Inter,sans-serif;font-size:14px;color:#ccc;line-height:1.75;margin:0;">{explanation}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("No explanation available.")
        except Exception as e:
            st.warning(f"Unavailable: {e}")

    # ── Difficulty ──
    with tab3:
        st.markdown("""<p style="font-family:Inter,sans-serif;font-size:13px;font-weight:700;color:#f5c542;margin:0 0 14px;text-transform:uppercase;letter-spacing:0.05em;">Difficulty</p>""", unsafe_allow_html=True)
        try:
            with st.spinner("Predicting difficulty..."):
                diff = predict_difficulty(
                    title=title,
                    description=description,
                    topics=patterns_string,
                )

            predicted  = diff['predicted_difficulty']
            confidence = diff['confidence']
            probs      = diff['probabilities']
            actual     = diff['leetcode_label']

            def difficulty_color(level):
                if level == "Easy":   return "#22c55e"
                if level == "Medium": return "#f5c542"
                if level == "Hard":   return "#ef4444"
                return "#888"

            pred_color   = difficulty_color(predicted)
            actual_color = difficulty_color(actual)
            easy_val     = probs.get('Easy', '0%')
            medium_val   = probs.get('Medium', '0%')
            hard_val     = probs.get('Hard', '0%')

            html = f"""<div style="background:#1e1e1e;border:0.5px solid #333;border-radius:8px;padding:20px 22px;margin-top:8px;">
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px;">
            <div style="background:#141414;border-radius:8px;padding:14px 16px;">
            <p style="font-family:Inter,sans-serif;font-size:12px;color:#888;margin:0 0 6px;">Predicted</p>
            <p style="font-family:Inter,sans-serif;font-size:24px;font-weight:800;color:{pred_color};margin:0;">{predicted}</p>
            </div>
            <div style="background:#141414;border-radius:8px;padding:14px 16px;">
            <p style="font-family:Inter,sans-serif;font-size:12px;color:#888;margin:0 0 6px;">Confidence</p>
            <p style="font-family:Inter,sans-serif;font-size:24px;font-weight:800;color:#fff;margin:0;">{confidence}</p>
            </div>
            <div style="background:#141414;border-radius:8px;padding:14px 16px;">
            <p style="font-family:Inter,sans-serif;font-size:12px;color:#888;margin:0 0 6px;">LeetCode Label</p>
            <p style="font-family:Inter,sans-serif;font-size:24px;font-weight:800;color:{actual_color};margin:0;">{actual}</p>
            </div>
            </div>
            <p style="font-family:Inter,sans-serif;font-size:12px;color:#888;margin:0 0 10px;text-transform:uppercase;letter-spacing:0.05em;">Probability Breakdown</p>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
            <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-family:Inter,sans-serif;font-size:13px;color:#22c55e;font-weight:600;">Easy</span>
            <span style="font-family:Inter,sans-serif;font-size:13px;color:#fff;font-weight:700;">{easy_val}</span>
            </div>
            <div style="background:#333;border-radius:999px;height:6px;">
            <div style="width:{easy_val};background:#22c55e;border-radius:999px;height:6px;"></div>
            </div>
            </div>
            <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-family:Inter,sans-serif;font-size:13px;color:#f5c542;font-weight:600;">Medium</span>
            <span style="font-family:Inter,sans-serif;font-size:13px;color:#fff;font-weight:700;">{medium_val}</span>
            </div>
            <div style="background:#333;border-radius:999px;height:6px;">
            <div style="width:{medium_val};background:#f5c542;border-radius:999px;height:6px;"></div>
            </div>
            </div>
            <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-family:Inter,sans-serif;font-size:13px;color:#ef4444;font-weight:600;">Hard</span>
            <span style="font-family:Inter,sans-serif;font-size:13px;color:#fff;font-weight:700;">{hard_val}</span>
            </div>
            <div style="background:#333;border-radius:999px;height:6px;">
            <div style="width:{hard_val};background:#ef4444;border-radius:999px;height:6px;"></div>
            </div>
            </div>
            </div>
            </div>"""
            st.markdown(html, unsafe_allow_html=True)

            if diff['mislabelled_flag']:
                st.warning(f"⚠️ {diff['note']}")

        except Exception as e:
            st.warning(f"Unavailable: {e}")

    # ── Companies ──
    with tab4:
        try:
            with st.spinner("Finding companies..."):
                result = find_companies(
                    title=title,
                    description=description,
                    pattern=patterns_string,
                )

            if 'error' in result:
                st.warning(result['error'])
            else:
                companies = result.get('companies', [])
                note      = result.get('note', '')

                if companies:
                    pills_html = "".join([
                        f'<span style="font-family:Inter,sans-serif;font-size:13px;font-weight:500;background:#1e1e1e;color:#fff;border:0.5px solid #333;border-radius:999px;padding:5px 14px;display:inline-block;margin:4px;">{c}</span>'
                        for c in companies
                    ])
                    html = f"""<div style="background:#1e1e1e;border:0.5px solid #333;border-radius:8px;padding:20px 22px;margin-top:8px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
                    <p style="font-family:Inter,sans-serif;font-size:13px;font-weight:700;color:#f5c542;margin:0;text-transform:uppercase;letter-spacing:0.05em;">Companies</p>
                    <span style="font-family:Inter,sans-serif;font-size:12px;color:#888;background:#141414;border-radius:999px;padding:3px 12px;">{len(companies)} companies</span>
                    </div>
                    <p style="font-family:Inter,sans-serif;font-size:12px;color:#888;margin:0 0 12px;">{note}</p>
                    <div style="display:flex;flex-wrap:wrap;gap:4px;">{pills_html}</div>
                    </div>"""
                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.info("No companies found.")
        except Exception as e:
            st.warning(f"Unavailable: {e}")

    # ── Similar Problems ──
    with tab5:
        st.markdown("""<p style="font-family:Inter,sans-serif;font-size:13px;font-weight:700;color:#f5c542;margin:0 0 14px;text-transform:uppercase;letter-spacing:0.05em;">Similar Problems</p>""", unsafe_allow_html=True)
        try:
            with st.spinner("Finding similar problems..."):
                similar = predict_similar(
                    title=title,
                    description=description,
                    topics=patterns_string,
                    top_k=10,
                )

            if similar:
                def diff_color(level):
                    if level == "Easy":   return "#22c55e"
                    if level == "Medium": return "#f5c542"
                    if level == "Hard":   return "#ef4444"
                    return "#888"

                for i, p in enumerate(similar, 1):
                    d_color = diff_color(p['difficulty'])
                    html = f"""<div style="background:#1e1e1e;border:0.5px solid #333;border-radius:8px;padding:14px 18px;margin-bottom:10px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                    <span style="font-family:Inter,sans-serif;font-size:14px;font-weight:600;color:#fff;">{i}. {p['title']}</span>
                    <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-family:Inter,sans-serif;font-size:12px;font-weight:700;color:{d_color};background:{d_color}20;padding:3px 10px;border-radius:999px;">{p['difficulty']}</span>
                    <span style="font-family:Inter,sans-serif;font-size:12px;color:#888;">Score: <span style="color:#fff;font-weight:600;">{p['similarity_score']}</span></span>
                    </div>
                    </div>
                    <span style="font-family:Inter,sans-serif;font-size:12px;color:#888;background:#141414;padding:3px 10px;border-radius:999px;">{p['pattern']}</span>
                    </div>"""
                    st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No similar problems found.")
        except Exception as e:
            st.warning(f"Unavailable: {e}")