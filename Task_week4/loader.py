"""File 1 - loader.py
Turns uploaded files (txt, md, py, pdf, wav, mp3) into LangChain Documents.
Audio files are transcribed to text with faster-whisper.
"""
import os
import tempfile

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from faster_whisper import WhisperModel

_whisper = None  


def _get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel("base", device="cpu")
    return _whisper


def load_files(files):
    """files = list of Streamlit UploadedFile objects. Returns a list of Documents."""
    all_docs = []

    for file in files: 
        ext = file.name.split(".")[-1].lower()

   
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.getvalue())
            path = tmp.name

        if ext in ("wav", "mp3"):
            segments, _ = _get_whisper().transcribe(path)
            text = " ".join(segment.text for segment in segments)
            all_docs.append(Document(page_content=text, metadata={"source": file.name}))
        else:
            if ext == "pdf":
                docs = PyPDFLoader(path).load()
            else:
                docs = TextLoader(path, encoding="utf-8").load()
            for d in docs:
                d.metadata["source"] = file.name  
            all_docs.extend(docs)

        os.remove(path)

    return all_docs