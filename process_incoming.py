import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import joblib 
import requests


OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_available(timeout_s: int = 2) -> bool:
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout_s)
        return True
    except Exception:
        return False


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post(f"{OLLAMA_BASE_URL}/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    }, timeout=120)

    r.raise_for_status()
    embedding = r.json()["embeddings"]
    return embedding


def inference(prompt):
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }, timeout=300)

    r.raise_for_status()
    return r.json()


def _rank_chunks_fallback_tfidf(df: pd.DataFrame, query: str, top_k: int = 5) -> pd.DataFrame:
    texts = df["text"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    q = vectorizer.transform([query])
    sims = cosine_similarity(X, q).flatten()
    max_indx = sims.argsort()[::-1][0:top_k]
    return df.iloc[max_indx]


def _templated_answer(df_top: pd.DataFrame, query: str) -> str:
    lines = [
        "I can help only with questions related to the course videos.",
        "Here are the closest matching parts from the subtitles:",
        "",
    ]
    for _, row in df_top.iterrows():
        number = row.get("number", "")
        title = row.get("title", "")
        start = row.get("start", "")
        end = row.get("end", "")
        text = str(row.get("text", "")).strip().replace("\n", " ")
        if len(text) > 240:
            text = text[:240] + "…"
        lines.append(f"- Video {number} — {title} ({start}s to {end}s): {text}")

    lines += [
        "",
        "If you install/start Ollama with models `bge-m3` and `llama3.2`, I can generate a more natural, guided answer with timestamps.",
    ]
    return "\n".join(lines)

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
top_results = 5

use_ollama = _ollama_available()

if use_ollama:
    question_embedding = create_embedding([incoming_query])[0]
    similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.loc[max_indx]
else:
    new_df = _rank_chunks_fallback_tfidf(df, incoming_query, top_k=top_results)

prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

if use_ollama:
    response = inference(prompt)["response"]
else:
    response = _templated_answer(new_df, incoming_query)

print(response)

with open("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])