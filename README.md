# 🎓 RAG-Based AI Teaching Assistant
Live- https://teachingaiassistant.streamlit.app/

An AI-powered Retrieval-Augmented Generation (RAG) Teaching Assistant built using **Python, Streamlit, TF-IDF, Ollama, and Vector Embeddings**.

This project allows users to ask questions related to course videos.  
The assistant retrieves the most relevant subtitle chunks from the course content and optionally generates intelligent answers using local LLMs via Ollama.
<img width="1536" height="666" alt="image" src="https://github.com/user-attachments/assets/5941a08d-1e32-4a66-9280-308ad68e846b" />


---

# 🚀 Features

- 📹 Works on your own course videos
- 🎤 Converts videos → audio → subtitle JSON
- 🧠 Embedding-based semantic search
- 🔍 TF-IDF fallback retrieval mode
- 🤖 Optional Ollama LLM integration
- ⚡ Streamlit-based interactive UI
- 🕒 Timestamp-based video navigation
- 💾 Precomputed embeddings using Joblib
- 📚 RAG pipeline implementation

---

# 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Ollama
- TF-IDF Vectorizer
- Cosine Similarity

---

# 📂 Project Structure

```bash
.
├── jsons/
├── embeddings.joblib
├── mp3_to_json.py
├── preprocess_json.py
├── process_incoming.py
├── streamlit_app.py
├── video_to_mp3.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Anshul-Rajpoot/RAG-Based-AI-Teaching-Assistant-.git
cd rag-based-ai-teaching-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## Run CLI Version

```bash
python process_incoming.py
```

---

# 🤖 Optional Ollama Setup

If Ollama is installed and running locally, the assistant can generate natural AI answers.

Install Ollama:

```bash
https://ollama.com/download
```

Pull required models:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

Start Ollama:

```bash
ollama serve
```

Default Ollama URL:

```bash
http://localhost:11434
```

---

# 🔄 How the RAG Pipeline Works

## Step 1 — Collect Course Videos

Move all course videos into your videos folder.

Example:

```bash
videos/
```

---

## Step 2 — Convert Videos to MP3

Run:

```bash
python video_to_mp3.py
```

This extracts audio from all videos.

---

## Step 3 — Convert MP3 to JSON

Run:

```bash
python mp3_to_json.py
```

This converts audio into subtitle/transcript JSON files.

---

## Step 4 — Generate Embeddings

Run:

```bash
python preprocess_json.py
```

This:

- Reads transcript JSON files
- Creates text chunks
- Generates embeddings
- Saves everything into:

```bash
embeddings.joblib
```

---

## Step 5 — Retrieval + AI Answering

When the user asks a question:

1. Relevant chunks are retrieved
2. Similarity search is performed
3. Top matching chunks are selected
4. Prompt is generated
5. LLM produces the final answer

---

# 🧠 Retrieval Modes

## 1. Ollama Semantic Search

Uses:
- `bge-m3` embeddings
- `llama3.2` LLM

Provides intelligent natural answers.

---

## 2. TF-IDF Fallback Mode

If Ollama is unavailable:

- Local TF-IDF retrieval is used
- Closest subtitle chunks are displayed

No GPU required.

---

# 💾 About embeddings.joblib

The project stores precomputed embeddings inside:

```bash
embeddings.joblib
```

Benefits:
- Faster startup
- No retraining needed
- Faster retrieval
- Easier deployment

---

# 🌐 Streamlit Deployment

This project can be deployed easily on:

- Streamlit Community Cloud
- Render
- Railway
- VPS

Recommended:

```bash
https://share.streamlit.io
```

---

# 📸 Example Use Cases

- AI course assistant
- YouTube lecture chatbot
- Educational search engine
- Video Q&A assistant
- Personal knowledge base

---

# 🧪 Example Questions

```text
Where is CSS Flexbox explained?

Which video teaches JavaScript arrays?

Where are HTML tables discussed?

Explain the CSS box model video timestamp.
```

---

# 📈 Future Improvements

- FAISS vector database
- OpenAI embeddings support
- YouTube auto-import
- Chat history memory
- Multi-course support
- PDF notes integration
- Voice query support

---

# 👨‍💻 Author

Anshul Rajpoot

Built as a practical implementation of:
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Local LLM Integration
- AI-powered educational assistants

---

# ⭐ If You Like This Project

Give it a star on GitHub ⭐
