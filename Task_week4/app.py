import streamlit as st
import json
import os
import assistant
import loader
import rag_DB

st.set_page_config(page_title="AI Coding Assistant", layout="wide")
st.title("AI Coding Assistant")

MEMORY_FILE = "memory.json"


def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"messages": [], "prefs": "Python"}


def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"messages": st.session_state.messages, "prefs": st.session_state.prefs},
            f, ensure_ascii=False, indent=2,
        )


saved = load_memory()
if "messages" not in st.session_state:
    st.session_state.messages = saved["messages"]
if "prefs" not in st.session_state:
    st.session_state.prefs = saved["prefs"]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


with st.sidebar:
    st.header("Settings")
    prefs = st.text_input("Preferences (language / framework)", key="prefs")

    st.header("File to explain")
    files = st.file_uploader(
        "Upload a file, then ask me to explain it",
        accept_multiple_files=True,
        type=["txt", "md", "py", "pdf", "wav", "mp3"],
    )
    if st.button("Load HumanEval dataset"):
            with st.spinner("Downloading and embedding 164 problems..."):
                n = rag_DB.add_humaneval()
            st.success(f"Added {n} problems to Chroma" if n else "HumanEval is already loaded")
        
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        save_memory()
        st.rerun()

    with st.expander("Memory (what the LLM sees)"):
        st.text(assistant.format_memory(st.session_state.messages))


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("route"):
            st.caption(f"Intent: {msg['route']}")
        st.markdown(msg["content"])  

      
        if msg["role"] == "assistant" and msg.get("route") == "GENERATE":
            code = assistant.extract_code(msg["content"])
            if code and st.button("Run code", key=f"run_{i}"):
                out, err = assistant.run_code(code)
                if out:
                    st.text("Output:")
                    st.code(out)
                if err:
                    st.text("Errors:")
                    st.code(err)
                if not out and not err:
                    st.info("The code ran with no output.")

query = st.chat_input("Ask me to generate or explain code...")

if query:
    memory = assistant.format_memory(st.session_state.messages)  
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if st.session_state.pending_question and query.strip().lower() == "skip":
                st.session_state.pending_question = None
                reply = "Okay, skipped. Ask me something new."
                route = "SKIP"
                st.markdown(reply)

        elif st.session_state.pending_question:

            rag_DB.add_solution(query, st.session_state.pending_question)
            st.session_state.pending_question = None
            reply = "Thanks! I stored your solution in my knowledge base. Ask me again and I'll use it."
            route = "LEARN"
            st.markdown(reply)


        else:
            route = assistant.classify(query)
            st.caption(f"Intent: {route}")
            llm_query = query
            if route == "EXPLAIN" and files:
                docs = loader.load_files(files)  
                file_text = "\n\n".join(d.page_content for d in docs)
                llm_query = f"{query}\n\nUploaded file content:\n{file_text[:12000]}"
            kind, output = assistant.respond(llm_query, route, memory, prefs)

            if kind == "stream":
                reply = st.write_stream(output)  
            else:  
                reply = output
                st.markdown(reply)
                st.session_state.pending_question = query

    st.session_state.messages.append({"role": "assistant", "content": reply, "route": route})
    save_memory()  
    st.rerun()