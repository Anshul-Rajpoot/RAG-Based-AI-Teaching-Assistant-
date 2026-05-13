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


def ollama_available(timeout_s: int = 2) -> bool:
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout_s)
        return True
    except Exception:
        return False


def create_embedding_ollama(text_list: list[str]) -> list[list[float]]:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": "bge-m3", "input": text_list},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def generate_ollama(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def generate_ollama_stream(prompt: str) -> Iterable[str]:
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": True},
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                payload = json.loads(line)
            except Exception:
                continue

            token = payload.get("response", "")

            if token:
                yield token

            if payload.get("done") is True:
                break


@st.cache_resource
def load_embeddings_df() -> pd.DataFrame:
    df = joblib.load("embeddings.joblib")

    # Convert embeddings lists → numpy arrays
    if "embedding" in df.columns:
        df["embedding"] = df["embedding"].apply(np.array)

    return df


@st.cache_resource
def build_tfidf(texts_tuple):
    texts = list(texts_tuple)

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(texts)

    return vectorizer, X


def rank_chunks(
    df: pd.DataFrame,
    query: str,
    top_k: int,
    use_ollama: bool,
) -> pd.DataFrame:

    if use_ollama:
        question_embedding = create_embedding_ollama([query])[0]

        similarities = cosine_similarity(
            np.vstack(df["embedding"]),
            [question_embedding],
        ).flatten()

        indices = similarities.argsort()[::-1][:top_k]

        return df.iloc[indices]

    texts_tuple = tuple(
        df["text"].fillna("").astype(str).tolist()
    )

    vectorizer, X = build_tfidf(texts_tuple)

    q = vectorizer.transform([query])

    sims = cosine_similarity(X, q).flatten()

    indices = sims.argsort()[::-1][:top_k]

    return df.iloc[indices]


def build_prompt(chunks_df: pd.DataFrame, query: str) -> str:
    return f"""
I am teaching web development in my Sigma web development course.

Here are subtitle chunks from the course videos:

{chunks_df[['title', 'number', 'start', 'end', 'text']].to_json(orient='records')}

------------------------------------------------

User Question:
"{query}"

Answer naturally and guide the user:
- which video contains the topic
- approximate timestamp
- what is taught there

Do not mention JSON format.

If the question is unrelated to the course,
politely say you can only answer course-related questions.
"""


def templated_answer(chunks_df: pd.DataFrame) -> str:
    lines = [
        "Closest matching parts from the subtitles:",
        "",
    ]

    for _, row in chunks_df.iterrows():

        number = row.get("number", "")
        title = row.get("title", "")
        start = row.get("start", "")
        end = row.get("end", "")

        text = str(row.get("text", "")).strip().replace("\n", " ")

        if len(text) > 260:
            text = text[:260] + "..."

        lines.append(
            f"- Video {number} — {title} "
            f"({start}s to {end}s): {text}"
        )

    lines += [
        "",
        "To enable AI-generated answers, start Ollama and pull:",
        "- bge-m3",
        "- llama3.2",
    ]

    return "\n".join(lines)


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(
    page_title="RAG Teaching Assistant",
    layout="wide",
)

st.title("RAG-based AI Teaching Assistant")

st.caption(
    "Ask questions about your course videos."
)

col_left, col_right = st.columns(
    [2, 1],
    vertical_alignment="top",
)

with col_right:

    top_k = st.number_input(
        "Top matches",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    auto_ollama = st.checkbox(
        "Use Ollama if available",
        value=True,
    )

    stream_llm = st.checkbox(
        "Stream answer (Ollama)",
        value=True,
    )

with col_left:

    query = st.text_input(
        "Your question",
        placeholder="e.g., Where is the CSS box model explained?",
    )

    run = st.button(
        "Ask",
        type="primary",
        width="stretch",
    )


if run:

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        df = load_embeddings_df()

    except Exception as e:
        st.error(f"Failed to load embeddings.joblib: {e}")
        st.stop()

    use_ollama = auto_ollama and ollama_available()

    with st.status(
        "Retrieving relevant chunks...",
        expanded=False,
    ):

        top_df = rank_chunks(
            df,
            query.strip(),
            int(top_k),
            use_ollama=use_ollama,
        )

    st.subheader("Top Matching Chunks")

    display_df = top_df[
        ["number", "title", "start", "end", "text"]
    ].copy()

    display_df["text"] = (
        display_df["text"]
        .astype(str)
        .str.replace("\n", " ")
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    prompt = build_prompt(
        top_df,
        query.strip(),
    )

    with open(
        "prompt.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(prompt)

    st.subheader("Answer")

    if use_ollama:

        if stream_llm:

            placeholder = st.empty()

            buffer = ""

            for token in generate_ollama_stream(prompt):
                buffer += token
                placeholder.markdown(buffer)

            answer = buffer

        else:

            answer = generate_ollama(prompt)

            st.write(answer)

    else:

        answer = templated_answer(top_df)

        st.write(answer)

    with open(
        "response.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(answer)

    if not use_ollama:
        st.info(
            "Ollama not detected. Running in fallback retrieval mode."
        )
