from typing import Any, Dict, List

import streamlit as st

from backend.core import run_llm

# it will take langchain documents as input and list out all links in that document
def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)  # in streamLit we save all data in session_state.
        st.rerun()  # for clear chat, we remove all session_state and rerun the stremlit (render again)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt) # user input

    with st.chat_message("assistant"):
        try:    # for rag execution we put in try block to handle the failure case as well
            with st.spinner("Retrieving docs and generating answer…"):  # spineer till our llm generates answer
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))    # list of sources

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")   # sources as a list

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}    # when llm answer we respond with role: assistant
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
