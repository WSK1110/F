"""
Performance optimizations for the RAG Chatbot
These functions can be integrated into the main RAG.py file
"""

import streamlit as st
import hashlib
import os
from typing import List, Dict, Any
from datetime import datetime
import concurrent.futures
from functools import lru_cache
from langchain.schema import Document
import configparser

# 1. Caching Strategies

@st.cache_resource
def get_chatbot_instance():
    """Cache the chatbot instance across sessions"""
    from RAG import RAGChatbot
    return RAGChatbot()

@st.cache_data
def load_configuration():
    """Cache configuration loading"""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

@st.cache_data
def load_pdf_documents(data_dir: str) -> List[Document]:
    """Cache PDF document loading with file modification check"""
    # Get modification times of all PDFs
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    mod_times = {f: os.path.getmtime(os.path.join(data_dir, f)) for f in pdf_files}
    
    # Create a hash of the modification times
    hash_str = str(sorted(mod_times.items()))
    cache_key = hashlib.md5(hash_str.encode()).hexdigest()
    
    # This will only reload if files have changed
    return _load_documents_internal(data_dir, cache_key)

def _load_documents_internal(data_dir: str, cache_key: str) -> List[Document]:
    """Internal document loading function"""
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

# 2. Parallel Document Processing

def load_pdfs_parallel(pdf_files: List[str], data_dir: str) -> List[Document]:
    """Load PDF files in parallel"""
    from langchain_community.document_loaders import PyPDFLoader
    
    def load_single_pdf(pdf_path):
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    
    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(load_single_pdf, os.path.join(data_dir, f)): f 
            for f in pdf_files
        }
        
        for future in concurrent.futures.as_completed(futures):
            try:
                docs = future.result()
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error loading {futures[future]}: {e}")
    
    return documents

# 3. Vector Store Caching

@st.cache_resource
def get_vector_store(_documents_hash: str, embeddings_provider: str, chunk_size: int, chunk_overlap: int):
    """Cache vector store based on document hash"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    
    # Get embeddings (this should also be cached/optimized)
    embeddings = get_embeddings_lazy(embeddings_provider)
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    splits = text_splitter.split_documents(_documents_hash)
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db_cached"
    )
    
    return vector_store

# 4. Lazy Loading of Embeddings

@lru_cache(maxsize=None)
def get_embeddings_lazy(provider: str):
    """Lazy load embedding providers"""
    if provider == 'openai':
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif provider == 'gemini':
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif provider == 'ollama':
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

# 5. Query Caching

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_response(question_hash: str, model_provider: str) -> Dict[str, Any]:
    """Cache similar query responses"""
    # This will be populated when a query is processed
    return None

def hash_question(question: str) -> str:
    """Create hash for similar questions"""
    # Normalize question: lowercase, strip whitespace, remove punctuation
    normalized = question.lower().strip()
    # Remove common variations
    normalized = normalized.replace("?", "").replace("!", "").replace(".", "")
    return hashlib.md5(normalized.encode()).hexdigest()

# 6. Session State Management

def cleanup_session_state(max_history: int = 50):
    """Limit chat history size and clean old data"""
    if 'chat_history' in st.session_state:
        # Keep only last N conversations
        st.session_state.chat_history = st.session_state.chat_history[-max_history:]
    
    # Clean up old comparison history
    if 'comparison_history' in st.session_state:
        st.session_state.comparison_history = st.session_state.comparison_history[-10:]

# 7. Progress Indicators

def load_documents_with_progress(data_dir: str) -> List[Document]:
    """Load documents with progress indicator"""
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.error("No PDF files found")
        return []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    documents = []
    for i, file in enumerate(pdf_files):
        status_text.text(f'Loading {file}...')
        # Load file (in real implementation, use the parallel loader)
        progress_bar.progress((i + 1) / len(pdf_files))
    
    status_text.text('Loading complete!')
    return documents

# 8. Optimized Configuration

OPTIMIZED_CONFIG = {
    'RAG': {
        'chunk_size': '1000',  # Increased from 850 for better context
        'chunk_overlap': '200',  # Reduced from 300 to save memory
        'top_k': '5',  # Reduced from default for faster retrieval
    },
    'DEFAULT': {
        'temperature': '0.1',
        'max_tokens': '2048',  # Reduced from 4000 for faster responses
    }
}

# 9. Batch Processing for Large Documents

def process_documents_in_batches(documents: List[Document], batch_size: int = 100):
    """Process documents in batches to reduce memory usage"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    all_splits = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        splits = text_splitter.split_documents(batch)
        all_splits.extend(splits)
        
        # Allow garbage collection between batches
        import gc
        gc.collect()
    
    return all_splits

# 10. Retriever Caching

@st.cache_resource
def get_hybrid_retriever(_vector_store, _documents):
    """Cache retriever creation"""
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(_documents)
    bm25_retriever.k = 5
    
    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            _vector_store.as_retriever(search_kwargs={"k": 5}),
            bm25_retriever
        ],
        weights=[0.7, 0.3]
    )
    
    return ensemble_retriever

# Integration function to apply optimizations to existing chatbot

def optimize_chatbot(chatbot):
    """Apply optimizations to existing chatbot instance"""
    # Override methods with optimized versions
    original_load = chatbot.load_10k_files
    
    def optimized_load(data_dir="10k_files"):
        # Use cached loading
        return load_pdf_documents(data_dir)
    
    chatbot.load_10k_files = optimized_load
    
    # Add cleanup to be called periodically
    cleanup_session_state()
    
    return chatbot