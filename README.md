# AI Coding Assistant

An intelligent coding assistant built with **Streamlit**, **Chroma** (RAG) and a free **OpenRouter** LLM.
It explains code, generates code from a knowledge base, runs the generated code, and learns from user feedback.

Cellula Technologies NLP Internship - Project 2.

## Features

- **Intent classifier:** an LLM decides if the request is *Explain Code* or *Generate Code*.
- **Router:**
  - *Explain* goes straight to the LLM (no RAG, no vector search).
  - *Generate* goes through RAG retrieval, a relevance check, then the LLM.
- **RAG with Chroma:** chunking, embeddings (`all-MiniLM-L6-v2`), similarity search, top-k retrieval, with metadata and source.
- **Relevance check:** a second LLM call judges the retrieved context. If it is not relevant, the assistant asks the user for the correct solution instead of guessing.
- **Human feedback learning:** the solution the user sends is embedded and stored in Chroma.
- **Long-term memory:** chat history and preferences are saved in `memory.json` and sent to the LLM with each request.
- **Code runner tool:** runs generated Python code and shows stdout and stderr (Run button).
- **Streamlit UI:** chat, history, response streaming, syntax-highlighted code, file upload, memory viewer.
- **Audio loading:** `.wav` / `.mp3` files are transcribed with faster-whisper.

## Flow

```
User -> Intent Classifier
         |-- Explain  -> LLM -> Answer
         |-- Generate -> Chroma search -> Relevance check
                           |-- Relevant     -> LLM -> Code -> Run button
                           |-- Not relevant -> ask user for solution -> store in Chroma
```

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface, memory saving, feedback handling |
| `assistant.py` | LLM, classifier, router, relevance check, code runner |
| `rag_db.py` | Chroma database: chunking, embeddings, search, HumanEval loading |
| `loader.py` | Loads txt, md, py, pdf and audio files |
| `requirements.txt` | Python packages |

## Knowledge base

The RAG data is the [OpenAI HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) dataset (164 Python problems with solutions).
Click **Load HumanEval dataset** in the sidebar once to embed it into Chroma.

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac / Linux
   ```
2. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file next to `app.py` with your [OpenRouter](https://openrouter.ai) key:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```

## How to use

1. Click **Load HumanEval dataset** (first run only).
2. **Generate:** ask for a function, for example *"write a function that returns the sum of numbers from 1 to n"*, then press **Run code**.
3. **Explain:** paste code in the chat, or upload a file in the sidebar, then ask *"explain this code"*.
4. **Teach it:** if it says it couldn't find relevant knowledge, send the correct code as your next message. Ask again and it will use it. Type `skip` to cancel.

## Notes

- The model is set by `MODEL` in `assistant.py`. Use any `:free` model from OpenRouter. Free models change over time and have daily request limits.
- The code runner uses a subprocess with a timeout. It is **not** a secure sandbox, so only run code you trust.
- Do not commit `.env`, `memory.json` or `chroma_db/`.

## Deployment

On Streamlit Community Cloud, add `OPENROUTER_API_KEY` in **App settings -> Secrets**.
