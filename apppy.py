from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import streamlit as st
import os

DATA_DIR = "./data"
INDEX_DIR = "./storage"
LLM_MODEL_NAME = "llama3.2:latest"

embedding_name = "mixedbread-ai/mxbai-embed-large-v1"
embed_model = HuggingFaceEmbedding(model_name=embedding_name)

llm = Ollama(model=LLM_MODEL_NAME, temperature = 0.2 ,request_timeout=220.0)
Settings.llm = llm
Settings.embed_model = embed_model


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


def prepare_template():
    """
    Prepare a prompt template for the QA system.
    """
    text_qa_template_str = """
    Tu es un scribe médiéval qui utilise un style ancien et d'époque dans sa réponse. Tu es le gardien d'archive très anciennes. Tu réponds aux questions des voyageurs. En voici une : {query_str}
    Voilà tout ce que tu sais et que tu as archivé à ce sujet :
    --------
    {context_str}
    --------
    À partir de ces connaissances à toi, et uniquement à partir d'elles, réponds en français à la question.
    Écris une réponse verbeuse.
    """
    qa_template = PromptTemplate(text_qa_template_str)
    return qa_template



st.markdown("""
            <img src='https://upload.wikimedia.org/wikipedia/en/a/ae/The_library_of_babel_-_bookcover.jpg' style='display: block; margin-left: auto; margin-right: auto; width: 160px;'>
            
            <div style='text-align: center;'>
            <h1>Parler avec mon texte</h1>
            <h5>Discussion avec les archives</h5>
            </div>
            """
            , unsafe_allow_html=True)

# Initialize session state messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Oui ?"}]

# Capture user input and append it to session state messages
if prompt := st.chat_input("Que veux-tu savoir, humain ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

qa_template = prepare_template()
query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=2)

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Excellente question. Je vais aller voir ce que les archives disent."):
            response = query_engine.query(prompt)
        st.markdown(response.response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response.response})