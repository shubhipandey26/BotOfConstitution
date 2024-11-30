# -*- coding: utf-8 -*-
import os
import streamlit as st
import pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV") or "us-east1-gcp"

# Check if keys are loaded correctly
if not google_api_key or not pinecone_api_key:
    st.error("API keys are not loaded properly. Please check your .env file.")

# Set up LLM and embedding model
llm = Gemini(api_key=google_api_key)
embed_model = GeminiEmbedding(model_name="models/embedding-001")

# Configure settings for LLM and embeddings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024

# Initialize Pinecone client
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Function to load documents and initialize the index in Pinecone
def ingest_documents():
    # Load documents from the specified folder
    documents = SimpleDirectoryReader("data").load_data()

    index_name = "cbotindex"
    dimension = 768  # Ensure this matches the embedding model dimension
    try:
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=dimension)
    except Exception as e:
        st.error(f"Error creating Pinecone index: {str(e)}")
    
    pinecone_index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    st.success("Documents ingested successfully!")
    return index

# Function to initialize the chat engine
def initialize_app():
    if 'index' not in st.session_state:
        st.session_state.index = ingest_documents()

    return st.session_state.index.as_chat_engine()

# Streamlit App UI
st.title("Constitution Chatbot")
st.write("Ingest documents to the Pinecone index and interact with the Knowledge Agent.")

# Button to trigger document ingestion
if st.button("Ingest Documents"):
    st.session_state.index = ingest_documents()
    st.session_state.chat_engine = st.session_state.index.as_chat_engine()

if 'chat_engine' not in st.session_state:
    st.session_state.chat_engine = initialize_app()

# Initialize chat history if not present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]
    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").markdown(content)

# Query input box
text_input = st.text_input("Your Question:")

# Function to handle message processing
def process_message(user_input):
    try:
        response = st.session_state.chat_engine.chat(user_input)
        response_text = response.response

        st.chat_message("assistant").markdown(response_text)

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    except Exception as e:
        st.error(f"Error: {str(e)}")

if text_input:
    if text_input.lower() == "exit":
        st.write("Exiting the chat. Goodbye!")
    else:
        with st.spinner("Processing your request..."):
            process_message(text_input)

# Optional: Add button to clear chat history
if st.button("Clear Chat"):
    st.session_state.chat_history.clear()
    st.session_state.chat_engine = initialize_app()
