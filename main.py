import base64
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
UI_CONFIG_FILE = "./ui.json"

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


def load_ui_config(ui_config_file):
    """
    Load the UI configuration from an external JSON file.
    """
    with open(ui_config_file, "r") as f:
        ui_config = json.load(f)
    return ui_config


def load_logo_as_base64(logo_path):
    """
    Load an image and convert it to a base64 string.
    """
    with open(logo_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"


# Load UI conf, logo & messages
ui_config = load_ui_config(UI_CONFIG_FILE)
ui_messages = load_messages(MESSAGES_FILE)
logo_base64 = load_logo_as_base64(ui_config["logo_path"])

st.markdown(
    ui_config["html_template"].format(logo=logo_base64),
    unsafe_allow_html=True,
)

# Initialize session state messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": ui_messages["greeting"]}
    ]


def get_conversation_history():
    """
    Prepare the conversation history to provide context to the model.
    """
    history = ""
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        history += f"{role}: {message['content']}\n"
    return history


# Capture user input and append it to session state messages
if prompt := st.chat_input(ui_messages["user_input_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

qa_template = prepare_template(TEMPLATE_FILE)
query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=2)

# Add conversation history to context
conversation_history = get_conversation_history()
full_prompt = f"{conversation_history}User: {prompt}"


if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner(ui_messages["wait_spinner"]):
            response = query_engine.query(full_prompt)
        st.markdown(response.response, unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": response.response}
        )
