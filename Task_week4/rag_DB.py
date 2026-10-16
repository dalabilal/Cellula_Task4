"""File 2 - rag_db.py
Everything about the Chroma vector database:
chunking -> embeddings -> store -> similarity search (top-k).
"""
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

def add_humaneval():
    """Loads the HumanEval dataset into Chroma (only once)."""
    from datasets import load_dataset

   
    if db.get(where={"source": "humaneval"}, limit=1)["ids"]:
        return 0

    data = load_dataset("openai/openai_humaneval", split="test")

    docs = []
    for row in data:
     
        text = row["prompt"] + row["canonical_solution"]
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "humaneval",
                    "task_id": row["task_id"],
                    "entry_point": row["entry_point"],
                },
            )
        )

    db.add_documents(docs) 
    return len(docs)


splitter = RecursiveCharacterTextSplitter.from_language(
    Language.PYTHON, chunk_size=800, chunk_overlap=100
)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


db = Chroma(
    collection_name="code_knowledge",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)


def add_documents(docs):
    """Chunk the documents, embed them, store them in Chroma. Returns number of chunks."""
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    if chunks:
        db.add_documents(chunks) 
    return len(chunks)


def add_solution(solution_text, question):
    """Human-feedback learning: store a solution given by the user."""
    doc = Document(
        page_content=f"Question: {question}\n\nSolution:\n{solution_text}",
        metadata={"source": "user_feedback", "question": question},
    )
    db.add_documents(splitter.split_documents([doc]))


def search(query, k=3):
    """Top-k similarity search over HumanEval + user feedback only."""
    return db.similarity_search(
        query, k=k,
        filter={"source": {"$in": ["humaneval", "user_feedback"]}},
    )