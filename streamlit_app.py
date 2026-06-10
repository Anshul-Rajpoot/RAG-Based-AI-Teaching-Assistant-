
# ===============================
# PROFESSIONAL RAG TEACHING ASSISTANT
# ===============================

import json
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_BASE_URL = "http://localhost:11434"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sigma AI Learning Assistant",
    page_icon="🎓",
    layout="wide",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.main {
    background-color:#0E1117;
}

.block-container {
    padding-top:1rem;
}

.hero {
    text-align:center;
    padding:20px;
    margin-bottom:20px;
}

.metric-box {
    background:#1E293B;
    padding:15px;
    border-radius:12px;
}

.answer-box {
    background:#111827;
    border:1px solid #374151;
    padding:20px;
    border-radius:15px;
}

.result-card {
    background:#111827;
    border-left:5px solid #3B82F6;
    padding:15px;
    border-radius:10px;
    margin-bottom:10px;
}

.stButton button {
    width:100%;
    border-radius:10px;
    height:3rem;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.title("🎓 Sigma AI Assistant")


    with st.expander("👨‍💻 Developer", expanded=True):
        st.write("Anshul Rajpoot  (2311401168)")
        st.write("MANIT Bhopal")
        st.write("ECE Department")

    st.divider()

    top_k = st.slider(
        "Top Matches",
        1,
        20,
        5,
    )

    auto_ollama = st.checkbox(
        "Use Ollama",
        value=True,
    )

    stream_llm = st.checkbox(
        "Stream Response",
        value=True,
    )

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero">
<h1>🎓 Sigma AI Learning Assistant</h1>
<p>Ask questions about your Web Development Course Videos</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# OLLAMA HELPERS
# --------------------------------------------------
def ollama_available(timeout_s=2):

    try:
        requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=timeout_s
        )
        return True
    except:
        return False


def create_embedding_ollama(text_list):

    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model":"bge-m3",
            "input":text_list
        },
        timeout=120
    )

    r.raise_for_status()

    return r.json()["embeddings"]


def generate_ollama(prompt):

    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":"llama3.2",
            "prompt":prompt,
            "stream":False
        },
        timeout=300
    )

    r.raise_for_status()

    return r.json()["response"]


def generate_ollama_stream(prompt):

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":"llama3.2",
            "prompt":prompt,
            "stream":True
        },
        stream=True,
        timeout=300
    ) as r:

        for line in r.iter_lines(
            decode_unicode=True
        ):

            if not line:
                continue

            payload = json.loads(line)

            token = payload.get(
                "response",
                ""
            )

            if token:
                yield token

            if payload.get("done"):
                break

# --------------------------------------------------
# LOAD EMBEDDINGS
# --------------------------------------------------
@st.cache_resource
def load_embeddings_df():

    df = joblib.load(
        "embeddings.joblib"
    )

    if "embedding" in df.columns:
        df["embedding"] = (
            df["embedding"]
            .apply(np.array)
        )

    return df


@st.cache_resource
def build_tfidf(texts):

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(
        texts
    )

    return vectorizer, X

# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------
def rank_chunks(
    df,
    query,
    top_k,
    use_ollama
):

    if use_ollama:

        q_embedding = (
            create_embedding_ollama([query])[0]
        )

        sims = cosine_similarity(
            np.vstack(df["embedding"]),
            [q_embedding]
        ).flatten()

    else:

        texts = tuple(
            df["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        vectorizer, X = build_tfidf(
            texts
        )

        q = vectorizer.transform(
            [query]
        )

        sims = cosine_similarity(
            X,
            q
        ).flatten()

    idx = sims.argsort()[::-1][:top_k]

    return df.iloc[idx]

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
def build_prompt(chunks_df, query):

    return f"""
You are a helpful instructor.

Course Subtitle Chunks:

{chunks_df[['title','number','start','end','text']].to_json(orient='records')}

User Question:
{query}

Explain:
1. Which video contains the answer
2. Timestamp
3. Concept explanation

Answer naturally.
"""


# --------------------------------------------------
# TIME FORMATTER
# --------------------------------------------------
def seconds_to_hms(seconds):

    seconds = int(float(seconds))

    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60

    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    return f"{mins:02d}:{secs:02d}"

# --------------------------------------------------
# MAIN
# --------------------------------------------------

try:

    df = load_embeddings_df()

    use_ollama = (
        auto_ollama and
        ollama_available()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Videos",
        int(df["number"].nunique())
    )

    c2.metric(
        "Chunks",
        len(df)
    )

    c3.metric(
        "Mode",
        "Ollama" if use_ollama else "TF-IDF"
    )

except Exception as e:

    st.error(
        f"Cannot load embeddings.joblib\n{e}"
    )

    st.stop()

query = st.chat_input(
    "Ask anything about the Sigma Web Development Course..."
)

if query:

    with st.spinner("🔍 Searching videos..."):

        top_df = rank_chunks(
            df,
            query,
            top_k,
            use_ollama
        )

    st.subheader("🤖 AI Answer")

    prompt = build_prompt(
        top_df,
        query
    )

    if use_ollama:

        if stream_llm:

            placeholder = st.empty()
            response = ""

            for token in generate_ollama_stream(prompt):

                response += token

                placeholder.markdown(
                    f"""
                    <div class='answer-box'>
                    {response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            response = generate_ollama(prompt)

            st.markdown(
                f"""
                <div class='answer-box'>
                {response}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "🔍 Retrieval Mode Active — Ollama not detected."
        )

        st.markdown(
            """
            <div class='answer-box'>
            Showing the most relevant video segments below.
            Start Ollama to get AI-generated explanations.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📹 Relevant Video Segments")

    for _, row in top_df.iterrows():

        start_time = seconds_to_hms(row["start"])
        end_time = seconds_to_hms(row["end"])

        with st.expander(
            f"📹 Video {row['number']} - {row['title']}",
            expanded=True
        ):

            st.markdown(
                f"### 📹 Video {row['number']}"
            )

            st.markdown(
                f"**{row['title']}**"
            )

            st.caption(f"🕒 {start_time} - {end_time}")
            st.write(str(row["text"])[:250] + "...")
