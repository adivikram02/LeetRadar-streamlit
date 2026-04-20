import joblib
import time
from google import genai
from google.genai import types
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

# Model config
MODEL_NAME = 'gemini-2.5-flash-lite'
MAX_TOKENS = 120

# Rate limiting - stay under 15 RPM
SLEEP_BETWEEN_CALLS = 4.5
RETRY_LIMIT = 3
RETRY_SLEEP = 5.0

SYSTEM_PROMPT = """You are a concise DSA (Data Structures & Algorithms) tutor.
Explain in exactly 1-2 sentences why a specific algorithmic pattern
is the optimal approach for a given LeetCode problem.

Rules:
- Focus on structural signals: optimal substructure, overlapping subproblems,
sorted input, sliding window opportunity, O(1) lookup need, graph traversal, etc.
- Be specific to THIS problem — not a generic pattern description.
- Do NOT restate the pattern name as a definition.
- Output only the explanation. No bullet points, no preamble, no labels."""


@st.cache_resource(show_spinner=False)
def create_client():
    return genai.Client(api_key=api_key)


@st.cache_data(show_spinner=False)
def explain_pattern(title: str, description: str, pattern: str) -> str:
    """
    Call Gemini Flash-Lite to explain why pattern fits this problem.
    Returns explanation string, or '' after RETRY_LIMIT failures.
    """
    user_prompt = (
        f"Problem: {title}\n"
        f"Pattern: {pattern}\n"
        f"Description: {description[:800].strip()}\n\n"
        "Why does this pattern fit?"
    )

    client = create_client()

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=MAX_TOKENS,
                ),
            )
            return response.text.strip()

        except Exception as e:
            err = str(e)
            if '429' in err or 'quota' in err.lower() or 'rate' in err.lower():
                wait = RETRY_SLEEP * attempt
                print(f'[Rate limit] sleeping {wait}s (attempt {attempt}/{RETRY_LIMIT})...')
                time.sleep(wait)
            else:
                print(f'[API error] attempt {attempt}/{RETRY_LIMIT}: {e}')
                time.sleep(RETRY_SLEEP)

    print(f'[FAILED] exhausted retries for: "{title}"')
    return ''


@st.cache_resource(show_spinner=False)
def load_explanation_index() -> dict:
    _index_cache = joblib.load('./models/pattern_explanation.joblib')
    n = len(_index_cache['title_to_explanation'])
    print(f'Index loaded — {n} problems')
    return _index_cache


@st.cache_data(show_spinner=False)
def get_pattern_explanation(
    title      : str,
    description: str = '',
    pattern    : str = '',
) -> dict:
    key = title.strip().lower()

    _index_cache = load_explanation_index()

    if _index_cache and key in _index_cache['title_to_explanation']:
        expl = _index_cache['title_to_explanation'].get(key, '')
        if expl:
            return {
                'title'              : title,
                'pattern'            : _index_cache['title_to_pattern'].get(key, pattern),
                'difficulty'         : _index_cache['title_to_difficulty'].get(key, ''),
                'pattern_explanation': expl,
                'source'             : 'cache',
            }

    print(f'[Cache miss] generating live for: "{title}"')
    expl = explain_pattern(title, description, pattern)
    return {
        'title'              : title,
        'pattern'            : pattern,
        'difficulty'         : '',
        'pattern_explanation': expl,
        'source'             : 'live',
    }

# import joblib
# import time
# from google import genai
# from google.genai import types
# import streamlit as st
# import os
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.environ.get("GEMINI_API_KEY")

# # _index_cache: dict = {}

# # model 
# MODEL_NAME = 'gemini-2.5-flash-lite'
# MAX_TOKENS = 120

# # rate limiting - stay under 15 RPM
# SLEEP_BETWEEN_CALLS = 4.5   # 60s / 15 RPM = 4s, adding 0.5s buffer
# RETRY_LIMIT = 3
# RETRY_SLEEP = 5.0
# CHECKPOINT_EVERY = 50

# @st.cache_data(show_spinner=False)
# def create_client():
#     client = genai.Client(api_key=api_key)

#     SYSTEM_PROMPT = """You are a concise DSA (Data Structures & Algorithms) tutor.
#     Explain in exactly 1-2 sentences why a specific algorithmic pattern
#     is the optimal approach for a given LeetCode problem.

#     Rules:
#     - Focus on structural signals: optimal substructure, overlapping subproblems,
#     sorted input, sliding window opportunity, O(1) lookup need, graph traversal, etc.
#     - Be specific to THIS problem — not a generic pattern description.
#     - Do NOT restate the pattern name as a definition.
#     - Output only the explanation. No bullet points, no preamble, no labels."""

#     st.session_state.client = client
#     st.session_state.system_prompt = SYSTEM_PROMPT

# @st.cache_data(show_spinner=False)
# def explain_pattern(title: str, description: str, pattern: str) -> str:
#     """
#     Call Gemini Flash-Lite to explain why pattern fits this problem 
#     Returns everything string, or '' after RETRY_LIMIT failures
#     """
#     user_prompt = (
#         f"Problem: {title}\n"
#         f"Pattern: {pattern}\n"
#         f"Description: {description[:800].strip()}\n\n"
#         "Why does this pattern fit?"
#     )
    
#     if "client" not in st.session_state or "system_prompt" not in st.session_state:
#         create_client()
    
#     client = st.session_state.client
#     SYSTEM_PROMPT = st.session_state.system_prompt

#     for attempt in range(1, RETRY_LIMIT + 1):
#         try:
#             response = client.models.generate_content(
#                 model=MODEL_NAME,
#                 contents=user_prompt,
#                 config=types.GenerateContentConfig(
#                     system_instruction=SYSTEM_PROMPT,
#                     max_output_tokens=MAX_TOKENS,
#                 ),
#             )
#             return response.text.strip()
        
#         except Exception as e:
#             err = str(e)
#             if '429' in err or 'quota' in err.lower() or 'rate' in err.lower():
#                 wait = RETRY_SLEEP * attempt
#                 print(f' [Rate limit] sleeping {wait}s (attempt {attempt}/{RETRY_LIMIT})...')
#                 time.sleep(wait)
#             else:
#                 print(f'[API error] attempt {attempt}/{RETRY_LIMIT}: {e}')
#                 time.sleep(RETRY_SLEEP)
    
#     print(f'[FAILED] exhausted retries for: "{title}"')
#     return ''

# @st.cache_resource(show_spinner=False)
# def load_explanation_index() -> dict:
#     _index_cache = joblib.load('./models/pattern_explanation.joblib')
#     n = len(_index_cache['title_to_explanation'])
#     # print(f'Index loaded — {n} problems')
#     return _index_cache

# @st.cache_data(show_spinner=False)
# def get_pattern_explanation(
#     title      : str,
#     description: str = '',
#     pattern    : str = '',
# ) -> dict:
#     key = title.strip().lower()
        
#     # _index_cache = load_explanation_index()

#     if _index_cache and key in _index_cache['title_to_explanation']:
#         expl = _index_cache['title_to_explanation'].get(key, '')
#         if expl:
#             return {
#                 'title'              : title,
#                 'pattern'            : _index_cache['title_to_pattern'].get(key, pattern),
#                 'difficulty'         : _index_cache['title_to_difficulty'].get(key, ''),
#                 'pattern_explanation': expl,
#                 'source'             : 'cache',
#             }
            
#     print(f'  [Cache miss] generating live for: "{title}"')
#     expl = explain_pattern(title, description, pattern)
#     return {
#         'title'              : title,
#         'pattern'            : pattern,
#         'difficulty'         : '',
#         'pattern_explanation': expl,
#         'source'             : 'live',
#     }

# _index_cache = load_explanation_index()

# # for row in df_out.sample(5, random_state=42).itertuples():
# #     result = get_pattern_explanation(row.title)
# #     print(f'  Title   : {result["title"]}')
# #     print(f'  Pattern : {result["pattern"]}')
# #     print(f'  Explain : {result["pattern_explanation"]}')
# #     print(f'  Source  : {result["source"]}')
# #     print()