import json
import os

import streamlit as st
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

DATA_DIR = "./data"
INDEX_DIR = "./storage"
LLM_MODEL_NAME = "llama-3.1-70b-versatile"
EMBEDDING_NAME = "mixedbread-ai/mxbai-embed-large-v1"
EMBED_MODEL = HuggingFaceEmbedding(model_name=EMBEDDING_NAME)
TEMPLATE_FILE = "./template.txt"
MESSAGES_FILE = "./messages.json"

llm = Groq(model=LLM_MODEL_NAME, temperature=0.2, request_timeout=220.0)
Settings.llm = llm
Settings.embed_model = EMBED_MODEL


@st.cache_data
def load_index():
    if not os.path.exists(INDEX_DIR):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=INDEX_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
    return index


index = load_index()


def prepare_template(template_file):
    """
    Load the prompt template
    """
    with open(template_file, "r") as f:
        text_qa_template_str = f.read()
    qa_template = PromptTemplate(text_qa_template_str)
    return qa_template


def load_messages(messages_file):
    """
    Load UI messages
    """
    with open(messages_file, "r") as f:
        messages = json.load(f)
    return messages


# Load the messages from the external JSON config file
ui_messages = load_messages(MESSAGES_FILE)

st.markdown(
    """
            <div style='text-align: center;'>
            <h1>Data ChatBot</h1>
            <h5>Ask anything to your own data</h5>
            </div>
            """,
    unsafe_allow_html=True,
)

# Initialize session state messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": ui_messages["greeting"]}
    ]

# Capture user input and append it to session state messages
if prompt := st.chat_input(ui_messages["user_input_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

qa_template = prepare_template(TEMPLATE_FILE)
query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=2)

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner(ui_messages["wait_spinner"]):
            response = query_engine.query(prompt)
        st.markdown(response.response, unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": response.response}
        )
