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

if "title" not in st.session_state:
    st.session_state.title = None
if "description" not in st.session_state:
    st.session_state.description = None


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

        # question not found — slug is invalid
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

if st.checkbox('Import from Leetcode'):
    with st.form("leetcode_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            string = st.text_input(
                label='slug',
                placeholder='Enter Leetcode Problem Name',
                label_visibility='collapsed'
            )
            slug = string_to_slug(string)
        with col2:
            import_button = st.form_submit_button("Import")

        if import_button:
            title, description = fetch_from_leetcode(slug)
            if title is None:
                st.error("Problem not found. Check the problem name and try again.")
            else:
                st.session_state.title = title
                st.session_state.description = description
                st.success("Fetched successfully")

else:
    title = st.text_input(label='Enter Problem Title')
    description = st.text_input(label='Enter Description')

    st.session_state.title = title
    st.session_state.description = description


# ── Analyze ────────────────────────────────────────────────────────────────────

analyze_clicked = st.button(label='Analyze')

if analyze_clicked and st.session_state.title and st.session_state.description:

    title       = st.session_state.title
    description = st.session_state.description

    # ── Patterns ──
    patterns        = predict_patterns(title, description)
    patterns_string = patterns_to_string(patterns)

    st.header("Analysis:")

    st.subheader("Patterns:")
    if patterns_string:
        st.text(patterns_string)
    else:
        st.text("No patterns detected.")

    st.divider()

    # ── Explanation ──
    st.subheader("Explanation:")
    try:
        explanation_result = get_pattern_explanation(
            title=title.lower(),
            description=description,
            pattern=patterns_string
        )
        explanation = explanation_result.get('pattern_explanation', '')
        if explanation:
            st.text(explanation)
        else:
            st.text("No explanation available.")
    except Exception as e:
        st.warning(f"Explanation unavailable: {e}")

    st.divider()

    # ── Companies ──
    st.subheader("Companies:")
    try:
        company_result = find_companies(
            title=title,
            description=description,
            pattern=patterns_string,
        )

        if 'error' in company_result:
            st.warning(company_result['error'])
        else:
            st.caption(company_result.get('note', ''))
            companies = company_result.get('companies', [])
            if companies:
                cols = st.columns(4)
                for i, company in enumerate(companies):
                    cols[i % 4].markdown(f"- {company}")
            else:
                st.text("No companies found.")
    except Exception as e:
        st.warning(f"Company prediction unavailable: {e}")

    st.divider()
    # ── Difficulty ──
    st.subheader("Difficulty:")
    try:
        diff_result = predict_difficulty(
            title=title,
            description=description,
            topics=patterns_string,
        )
        predicted    = diff_result['predicted_difficulty']
        confidence   = diff_result['confidence']
        probs        = diff_result['probabilities']
        actual_label = diff_result['leetcode_label']

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Predicted Label:** {predicted}")
        with col2:
            st.markdown(f"**Actual LeetCode Label:** {actual_label}")

        st.markdown(f"**Prediction confidence:** {confidence}")
        st.markdown(f"**Probability breakdown:** Easy: {probs.get('Easy', 'N/A')} | Medium: {probs.get('Medium', 'N/A')} | Hard: {probs.get('Hard', 'N/A')}")

        if diff_result['mislabelled_flag']:
            st.warning(f"⚠️ {diff_result['note']}")

    except Exception as e:
        st.warning(f"Difficulty prediction unavailable: {e}")

    st.divider()
    # ── Similar Problems ──
    st.subheader("Similar Problems:")
    try:
        similar = predict_similar(
            title=title,
            description=description,
            topics=patterns_string,
            top_k=10,
        )
        if similar:
            for i, prob in enumerate(similar, 1):
                st.markdown(f"**{i}. {prob['title']}** — {prob['difficulty']} | {prob['pattern']} | Score: {prob['similarity_score']}")
        else:
            st.text("No similar problems found.")
    except Exception as e:
        st.warning(f"Similar problem finder unavailable: {e}")




# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# from multi_pattern_classification import predict_patterns
# from pattern_explanation import get_pattern_explanation
# from company_matching import find_companies

# st.set_page_config(
#     # Title and icon for the browser's tab bar:
#     page_title="LeetRadar",
#     page_icon="🎯",
#     # Make the content take up the width of the page:
#     layout="wide",
# )

# import requests
# from bs4 import BeautifulSoup

# # if "result" not in st.session_state:
# #     st.session_state["result"] = None

# if "title" not in st.session_state:
#     st.session_state.title = None
# if "description" not in st.session_state:
#     st.session_state.description = None

# def string_to_slug(string: str):
#     string_list = string.lower().split(" ")
#     slug = "-".join(string_list)
#     return slug

# def patterns_to_string(patterns: list):
#     length = len(patterns)

#     if length == 0:
#         return ""
#     if length == 1:
#         return str(patterns[0][0])
#     if length == 2:
#         return f"{patterns[0][0]} and {patterns[1][0]}"
    
#     prefix = ", ".join(map(str, patterns[:-1][0]))
#     return f"{prefix} and {patterns[-1]}"

# @st.cache_data
# def fetch_from_leetcode(slug: str):
#     try:
#         url = "https://leetcode.com/graphql"
#         headers = {"Content-Type": "application/json"}

#         query = """
#         query getQuestion($titleSlug: String!) {
#             question(titleSlug: $titleSlug) {
#                 title
#                 content
#             }
#         }
#         """

#         resp = requests.post(
#             url,
#             json={"query": query, "variables": {"titleSlug": slug}},
#             headers=headers,
#             timeout=10
#         )

#         data = resp.json()["data"]["question"]

#         title = data["title"]
#         content = data["content"]

#         if not content:
#             return {"title": title, "description": ""}

#         # Strip HTML
#         description = BeautifulSoup(content, "html.parser") \
#                         .get_text(separator=" ") \
#                         .strip()

#         return title,description
        
#     except Exception as e:
#         print(f"Failed for {slug}: {e}")
#         st.warning(f"Failed for {slug}: {e}")
#         return {"title": "", "description": ""}
    
# def analyze(title,description):
#     patterns = predict_patterns(title,description)
#     patterns

# if st.checkbox('Import from Leetcode'):
#     with st.form("leetcode_form"):
#         col1, col2 = st.columns([4, 1])
#         with col1:
#             string = st.text_input(label='slug',placeholder='Enter Leetcode Problem Name',label_visibility='collapsed')
#             slug = string_to_slug(string)
#         with col2:
#             import_button = st.form_submit_button("Import")
        
#         if import_button:
#             # with st.spinner("Importing problem from Leetcode..."):
#             title, description = fetch_from_leetcode(slug)
#             st.session_state.title = title
#             st.session_state.description = description
#             st.success("Fetched successfully")
#             # print(title)
#             # print(description)
#                 # st.session_state.result = fetch_from_leetcode(slug)
            

# else:
#     title = st.text_input(label='Enter Problem Title')
#     description = st.text_input(label='Enter Description')

#     st.session_state.title = title
#     st.session_state.description = description

# analyze_clicked = st.button(label='Analyze')

# if analyze_clicked and st.session_state.title and st.session_state.description:
#     title = st.session_state.title
#     description = st.session_state.description
    
#     patterns = predict_patterns(title, description)

#     st.header("Analysis:")
#     st.subheader("Patterns:")
#     st.text("The patterns associated with this problem are")
#     patterns_string = patterns_to_string(patterns)
#     st.text(patterns_string)
#     st.divider()

#     st.subheader("Explanation:")
#     explanation = get_pattern_explanation(title=title.lower(), description=description, pattern=patterns_string)
#     explanation = explanation['pattern_explanation']
#     st.text(explanation)
#     st.divider()

#     st.subheader("Companies:")
#     company_result = find_companies(
#         title=title,
#         description=description,
#         pattern=patterns_string,
#     )

#     if 'error' in company_result:
#         st.warning(company_result['error'])
#     else:
#         st.caption(company_result['note'])
#         if company_result['companies']:
#             cols = st.columns(4)
#             for i, company in enumerate(company_result['companies']):
#                 cols[i % 4].markdown(f"- {company}")
#         else:
#             st.text("No companies found.")

        