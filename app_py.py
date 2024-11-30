import os
import subprocess
import sys

# Install missing dependencies
try:
    import pinecone
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pinecone-client==2.0.2"])

# Required imports
import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index import set_global_settings
from dotenv import load_dotenv
import pinecone

# Load environment variables
load_dotenv()

# API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "us-east1-gcp")

# Check API keys
if not openai_api_key or not pinecone_api_key:
    st.error("API keys for OpenAI or Pinecone are missing! Please check the .env file.")
    st.stop()

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Set up LLM and embedding model
llm = OpenAI(api_key=openai_api_key)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")  # Specify embedding model

# Configure settings globally
set_global_settings(llm=llm, embed_model=embed_model, chunk_size=1024)

# Function to ingest documents
def ingest_documents():
    try:
        # Load documents from a "data" folder
        documents = SimpleDirectoryReader("data").load_data()

        # Pinecone index name and dimension
        index_name = "cbotindex"
        dimension = 1536  # Dimension must match embedding model

        # Check if index exists, create if not
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

# Initialize chat engine
def initialize_app():
    if 'index' not in st.session_state:
        st.session_state.index = ingest_documents()

    # Create and return chat engine
    return st.session_state.index.as_chat_engine() if st.session_state.index else None

# Streamlit UI
st.title("Constitution Chatbot")
st.write("Ingest documents to Pinecone and interact with the Knowledge Agent.")

# Ingest documents button
if st.button("Ingest Documents"):
    st.session_state.index = ingest_documents()
    if st.session_state.index:
        st.session_state.chat_engine = st.session_state.index.as_chat_engine()

# Initialize chat engine once
if 'chat_engine' not in st.session_state:
    st.session_state.chat_engine = initialize_app()

# Initialize chat history
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

            # Display response
            st.chat_message("assistant").markdown(response_text)

            # Update chat history
            st.session_state.chat_history.append({"role": "user", "content": text_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Error: {e}")

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history = []  # Clear history
    st.experimental_rerun()
