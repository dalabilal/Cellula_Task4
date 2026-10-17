import os
import re
import subprocess
import sys
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import rag_DB

load_dotenv()  # reads the .env file next to the code

MODEL = "openrouter/free"

API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found. Put it in a .env file next to app.py")

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.2,
)
NO_CONTEXT_MSG = (
    "I couldn't find relevant knowledge for your request. Could you provide the "
    "correct solution or reference so I can learn it for future interactions?"
)


def ask(system, user):
    """One normal (non-streaming) LLM call."""
    return llm.invoke([("system", system), ("human", user)]).content


def stream(system, user):
    """Streaming LLM call: yields text pieces."""
    for chunk in llm.stream([("system", system), ("human", user)]):
        yield chunk.content


# ---------------------------------------------------------------- memory
def format_memory(messages, last_n=6):
    """Turns the last chat messages into text that we give to the LLM each time."""
    lines = []
    for m in messages[-last_n:]:
        lines.append(f"{m['role']}: {m['content'][:1500]}")
    return "\n".join(lines) if lines else "(empty)"


# ---------------------------------------------------------------- 1) classifier
def classify(query):
    system = (
        "Classify the user's request into exactly one class:\n"
        "EXPLAIN  = the user wants existing code explained, reviewed or debugged.\n"
        "GENERATE = the user wants new code to be written.\n"
        "Answer with ONE word only: EXPLAIN or GENERATE."
    )
    answer = ask(system, query).upper()
    return "EXPLAIN" if "EXPLAIN" in answer else "GENERATE"


# ---------------------------------------------------------------- 2) relevance check
def is_relevant(query, context):
    system = (
        "You are a relevance evaluator. Is the retrieved context sufficiently relevant "
        "to answer the user's programming request? It must contain the same kind of task "
        "or concept the user asks for. Using the same programming language is NOT enough. "
        "Answer with ONE word only: RELEVANT or NOT_RELEVANT."
    )
       
    answer = ask(system, f"Request:\n{query}\n\nRetrieved context:\n{context}")
    print("RELEVANCE CHECK ANSWER:", answer)  # visible in your terminal
    return "NOT_RELEVANT" not in answer.upper().replace(" ", "_")


# ---------------------------------------------------------------- 3) router
def respond(query, route, memory, prefs):
    """Returns (kind, output):
    kind = "stream"     -> output is a generator of text (show it with st.write_stream)
    kind = "no_context" -> output is the 'please teach me' message
    """
    
    # Route 1: EXPLAIN  -> LLM only. NO retrieval, NO vector DB, NO embeddings.
    if route == "EXPLAIN":
        system = (
            "You are a coding assistant. Explain the code the user provides clearly "
            "and simply. If asked, go line by line. Point out bugs if you see any.\n"
            f"User preferences: {prefs}\nConversation so far:\n{memory}"
        )
        return "stream", stream(system, query)

    # Route 2: GENERATE -> RAG retrieval -> relevance check -> LLM
    docs = rag_DB.search(query, k=3)
    print("RETRIEVED:", [d.metadata.get("task_id", d.metadata.get("source")) for d in docs])
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    if not docs or not is_relevant(query, context):
        return "no_context", NO_CONTEXT_MSG  # do not hallucinate

    system = (
        "You are an expert coding assistant. Write code using the retrieved knowledge.\n"
        "Your answer must contain: complete source code, comments, best practices and "
        "error handling. Put the code in a Markdown code block (```python). "
        "A short explanation is optional.\n"
        f"User preferences: {prefs}\n"
        f"Conversation so far:\n{memory}\n\n"
        f"Retrieved knowledge:\n{context}"
    )
    return "stream", stream(system, query)


# ---------------------------------------------------------------- code runner tool
def extract_code(text):
    """Gets the first ```python block from an answer (or None)."""
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    return blocks[0] if blocks else None


def run_code(code, timeout=10):
    """Code Runner Tool: receives code as a string, runs it, returns (stdout, stderr).
    The LLM never runs code itself - the app calls this tool when the user clicks 'Run'.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=timeout
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", f"Stopped: the code ran longer than {timeout} seconds."
    finally:
        os.remove(path)