# How to use this RAG AI Teaching assistant on your own data

## Quick start (runs even without Ollama)

Install deps:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Or run the CLI version:

```bash
python process_incoming.py
```

Note: If Ollama is running at `http://localhost:11434` with models `bge-m3` and `llama3.2`, the app will use it to generate a natural answer. If Ollama is not available, it automatically falls back to local retrieval and shows the closest matching chunks.

## Step 1 - Collect your videos
Move all your video files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by ruunning video_to_mp3

## Step 3 - Convert mp3 to json
Convert all the mp3 files to json by ruunning mp3_to_json

## Step 4 - Convert the json files to Vectors
Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to LLM

Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM
