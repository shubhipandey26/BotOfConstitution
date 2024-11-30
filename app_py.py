import os
import subprocess
import sys

# Install dependencies dynamically if missing
try:
    import pinecone
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pinecone-client>=2.0.2"])

import streamlit as st
from dotenv import load_dotenv
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.llms.openai import OpenAI
from llama_index import SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import pinecone

# Load environment variables from .env
load_dotenv()

# Get environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "us-east1-gcp")

# Check for API keys
if not openai_api_key or not pinecone_api_key:
    st.error("API keys for OpenAI or Pinecone are missing! Please check the .env file.")
    st.stop()

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Configure OpenAI LLM and embedding model
llm = OpenAI(api_key=openai_api_key)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Function to ingest documents and set up the Pinecone vector store
def ingest_documents():
    try:
        # Load documents from the `data` folder
        documents = SimpleDirectoryReader("data").load_data()

        # Check if the Pinecone index exists; if not, create one
        index_name = "cbotindex"
        dimension = 1536  # Match OpenAI embedding dimensions
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=dimension)

        # Connect to the Pinecone index
        pinecone_index = pinecone.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        # Create a storage context and initialize the index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        st.success("Documents successfully ingested!")
        return index
    except Exception as e:
        st.error(f"Error during document ingestion: {e}")
        return None

# Function to initialize the chat engine
def initialize_app():
    if "index" not in st.session_state:
        st.session_state.index = ingest_documents()

    # Set up chat engine
    return (
        st.session_state.index.as_chat_engine(chat_mode="simple")
        if st.session_state.index
        else None
    )

# Streamlit UI
st.title("Constitution Chatbot")
st.write("Ingest documents to the Pinecone index and interact with the bot.")

# Button to trigger document ingestion
if st.button("Ingest Documents"):
    st.session_state.index = ingest_documents()
    if st.session_state.index:
        st.session_state.chat_engine = st.session_state.index.as_chat_engine(chat_mode="simple")

# Initialize chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = initialize_app()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").markdown(content)

# Input box for user query
text_input = st.text_input("Your Question:")

if text_input:
    if text_input.lower() == "exit":
        st.write("Exiting the chat. Goodbye!")
    else:
        try:
            # Get response from chat engine
            response = st.session_state.chat_engine.chat(text_input)
            response_text = response.response

            # Display the response
            st.chat_message("assistant").markdown(response_text)

            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": text_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Add button to clear chat history
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
