import os
import subprocess
import sys
import streamlit as st
from dotenv import load_dotenv
import pinecone
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    PineconeVectorStore,
    Settings
)
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
import importlib

# Check and install pinecone-client if it's not installed
try:
    import pinecone
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pinecone-client==2.0.2"])

# Load environment variables from .env file for sensitive keys
load_dotenv()

# Ensure environment variables are loaded
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "us-east1-gcp")

# Check if the required environment variables are set
if not google_api_key or not pinecone_api_key:
    st.error("API keys for Google or Pinecone are missing! Please check the .env file.")
    st.stop()  # Stop execution if keys are missing

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
    try:
        # Load documents from the specified folder
        documents = SimpleDirectoryReader("data").load_data()

        # Check if the Pinecone index exists, if not create one
        index_name = "cbotindex"
        dimension = 768  # Ensure this matches the embedding model dimension
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=dimension)

        # Initialize Pinecone index and vector store
        pinecone_index = pinecone.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        # Create storage context and index from documents
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        st.success("Documents ingested successfully!")
        return index
    except Exception as e:
        st.error(f"Error during document ingestion: {e}")
        return None

# Function to initialize the chat engine
def initialize_app():
    # Ingest documents (if not already ingested)
    if 'index' not in st.session_state:
        st.session_state.index = ingest_documents()

    # Create and return the chat engine
    return st.session_state.index.as_chat_engine() if st.session_state.index else None

# Streamlit App UI
st.title("Constitution Chatbot")
st.write("Ingest documents to the Pinecone index and interact with the Knowledge Agent.")

# Button to trigger document ingestion
if st.button("Ingest Documents"):
    st.session_state.index = ingest_documents()
    if st.session_state.index:
        st.session_state.chat_engine = st.session_state.index.as_chat_engine()

# Initialize the chat engine only once
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

if text_input:
    if text_input.lower() == "exit":
        st.write("Exiting the chat. Goodbye!")
    else:
        try:
            # Get the response from the chat engine
            response = st.session_state.chat_engine.chat(text_input)
            response_text = response.response

            # Display the response
            st.chat_message("assistant").markdown(response_text)

            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": text_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Optional: Add button to clear chat history
if st.button("Clear Chat"):
    st.session_state.chat_history = []  # Clear the chat history
    st.experimental_rerun()  # Simulate a restart by re-running the script
