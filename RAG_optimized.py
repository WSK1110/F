import os
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
import re
import time
import hashlib
from contextlib import contextmanager
import psutil

# LLM and Embedding imports - will be loaded conditionally
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama

# Core RAG components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# LightRAG improvements
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.merger_retriever import MergerRetriever

# Document processing
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document

# Configuration
import configparser

# Performance monitoring
@contextmanager
def performance_monitor(operation_name: str):
    """Monitor performance of operations"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Store performance data in session state
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
        
        st.session_state.performance_metrics.append({
            'operation': operation_name,
            'duration': end_time - start_time,
            'memory_change': end_memory - start_memory,
            'timestamp': datetime.now().isoformat()
        })

# Constants for optimization
MAX_CHAT_HISTORY = 50
MAX_COMPARISON_HISTORY = 20
CACHE_TTL_DOCUMENTS = 3600  # 1 hour
CACHE_TTL_QUERIES = 300     # 5 minutes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=CACHE_TTL_DOCUMENTS, show_spinner="Loading environment configuration...")
def load_environment_from_config():
    """Load environment variables from config.ini - cached"""
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        env_vars = {}
        
        # Set Google API key
        if 'LLM' in config and 'google_api_key' in config['LLM']:
            google_key = config['LLM']['google_api_key'].strip()
            if google_key and google_key != '':
                env_vars['GOOGLE_API_KEY'] = google_key
                logger.info("Google API key loaded from config.ini")
        
        # Set OpenAI API key
        if 'LLM' in config and 'openai_api_key' in config['LLM']:
            openai_key = config['LLM']['openai_api_key'].strip()
            if openai_key and openai_key != '':
                env_vars['OPENAI_API_KEY'] = openai_key
                logger.info("OpenAI API key loaded from config.ini")
        
        # Apply to environment
        for key, value in env_vars.items():
            os.environ[key] = value
            
        return env_vars
                
    except Exception as e:
        logger.warning(f"Could not load environment from config: {e}")
        return {}

def get_documents_hash(data_dir: str) -> str:
    """Generate hash of document files for cache invalidation"""
    try:
        file_stats = []
        for file in os.listdir(data_dir):
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(data_dir, file)
                stat = os.stat(file_path)
                file_stats.append(f"{file}:{stat.st_mtime}:{stat.st_size}")
        
        combined = "|".join(sorted(file_stats))
        return hashlib.md5(combined.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate document hash: {e}")
        return str(time.time())  # Fallback to timestamp

def extract_company_name(filename):
    """Extract company name from filename"""
    match = re.match(r"([A-Za-z]+)", filename)
    return match.group(1) if match else "Unknown"

def get_all_companies(documents):
    """Get all unique companies from documents"""
    return sorted(set(doc.metadata.get('company', 'Unknown') for doc in documents))

def retrieve_per_company(vector_store, question, documents, k_per_company=2):
    """Retrieve documents per company for comparison queries"""
    companies = get_all_companies(documents)
    all_chunks = []
    for company in companies:
        company_chunks = vector_store.similarity_search(
            question, 
            k=k_per_company, 
            filter={'company': company}
        )
        if not company_chunks:
            company_chunks = [Document(
                page_content=f"Not mentioned in the provided documents for {company}.",
                metadata={'company': company}
            )]
        all_chunks.extend(company_chunks)
    return all_chunks

def is_comparison_question(question):
    """Check if question is asking for comparison"""
    return any(word in question.lower() for word in ['compare', 'versus', 'difference', 'both', 'all companies'])

@st.cache_data(ttl=CACHE_TTL_DOCUMENTS, show_spinner="Loading documents...")
def load_10k_files_cached(data_dir: str = "10k_files") -> List[Document]:
    """Load 10-K PDF files from directory - cached version"""
    with performance_monitor("Document Loading"):
        documents = []
        
        if not os.path.exists(data_dir):
            st.warning(f"Data directory {data_dir} not found.")
            return documents
        
        # List all PDF files
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            st.error("No PDF files found in the 10k_files directory")
            return documents
        
        # Load PDF files
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        try:
            documents = loader.load()
            
            # Tag each document with company name
            for doc in documents:
                filename = doc.metadata.get('source', '')
                company = extract_company_name(os.path.basename(filename))
                doc.metadata['company'] = company
            
            logger.info(f"Successfully loaded {len(documents)} documents from {data_dir}")
            
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            logger.error(f"Document loading error: {e}")
        
        return documents

@st.cache_data(ttl=CACHE_TTL_DOCUMENTS, show_spinner="Processing document chunks...")
def create_document_chunks_cached(documents: List[Document], chunk_size: int = 850, chunk_overlap: int = 300) -> List[Document]:
    """Create document chunks - cached version"""
    with performance_monitor("Document Chunking"):
        if not documents:
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} text chunks")
        return splits

class RAGChatbotOptimized:
    """
    Optimized RAG chatbot with comprehensive caching and performance improvements.
    """
    
    def __init__(self, config_file: str = "config.ini"):
        self.config = self._load_config(config_file)
        self.vector_store = None
        self.qa_chain = None
        
        # Initialize memory with size limit
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            max_token_limit=2000  # Limit memory size
        )
        
        # Initialize performance tracking
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
    
    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """Load configuration with caching consideration"""
        config = configparser.ConfigParser()
        config.read(config_file)

        # Set defaults (same as original but optimized)
        defaults = {
            'DEFAULT': {
                'chunk_size': '850',
                'chunk_overlap': '300',
                'temperature': '0.1',
                'max_tokens': '4000'
            },
            'LLM': {
                'provider': 'gemini',
                'model': 'models/gemini-1.5-pro',
                'api_key': ''
            },
            'EMBEDDINGS': {
                'provider': 'gemini',
                'model': 'models/embedding-001',
                'api_key': ''
            },
            'RAG': {
                'chunk_size': '850',
                'chunk_overlap': '300',
                'similarity_top_k': '8'
            },
            'LIGHT_RAG': {
                'use_hybrid_retrieval': 'true',
                'use_reranking': 'true',
                'use_compression': 'true',
                'sparse_weight': '0.50',
                'dense_weight': '0.80',
                'rerank_top_k': '12',
                'final_top_k': '3'
            }
        }

        # Apply defaults
        for section_name, section_data in defaults.items():
            if section_name not in config:
                config[section_name] = {}
            for key, value in section_data.items():
                config[section_name].setdefault(key, value)

        return config
    
    @st.cache_resource
    def get_llm_cached(_self, provider: str, model: str, api_key: str):
        """Get LLM instance - cached to avoid repeated initialization"""
        with performance_monitor(f"LLM Initialization ({provider})"):
            if provider == 'gemini':
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=float(_self.config['DEFAULT']['temperature']),
                    max_output_tokens=int(_self.config['DEFAULT']['max_tokens'])
                )
            
            elif provider == 'openai':
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model,
                    openai_api_key=api_key,
                    temperature=float(_self.config['DEFAULT']['temperature']),
                    max_tokens=int(_self.config['DEFAULT']['max_tokens'])
                )
            
            elif provider == 'ollama':
                from langchain_community.llms import Ollama
                return Ollama(
                    model=model,
                    temperature=float(_self.config['DEFAULT']['temperature'])
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @st.cache_resource
    def get_embeddings_cached(_self, provider: str, model: str, api_key: str):
        """Get embeddings instance - cached to avoid repeated initialization"""
        with performance_monitor(f"Embeddings Initialization ({provider})"):
            if provider == 'gemini':
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model=model,
                    google_api_key=api_key
                )
            
            elif provider == 'openai':
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    model=model,
                    openai_api_key=api_key
                )
            
            elif provider == 'ollama':
                from langchain_community.embeddings import OllamaEmbeddings
                return OllamaEmbeddings(model=model)
            
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
    
    def get_llm(self, provider: str = None, model: str = None):
        """Get LLM with caching"""
        provider = provider or self.config['LLM']['provider']
        model = model or self.config['LLM']['model']
        
        # Get API key
        if provider == 'gemini':
            api_key = self.config['LLM']['api_key'] or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                st.error("Google API key not found.")
                return None
        elif provider == 'openai':
            api_key = self.config['LLM']['api_key'] or os.getenv('OPENAI_API_KEY')
            if not api_key:
                st.error("OpenAI API key not found.")
                return None
        else:
            api_key = ""
        
        try:
            return self.get_llm_cached(provider, model, api_key)
        except Exception as e:
            st.error(f"Error initializing LLM: {e}")
            return None
    
    def get_embeddings(self, provider: str = None, model: str = None):
        """Get embeddings with caching"""
        provider = provider or self.config['EMBEDDINGS']['provider']
        model = model or self.config['EMBEDDINGS']['model']
        
        # Get API key
        if provider == 'gemini':
            api_key = self.config['EMBEDDINGS']['api_key'] or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                st.error("Google API key not found for embeddings.")
                return None
        elif provider == 'openai':
            api_key = self.config['EMBEDDINGS']['api_key'] or os.getenv('OPENAI_API_KEY')
            if not api_key:
                st.error("OpenAI API key not found for embeddings.")
                return None
        else:
            api_key = ""
        
        try:
            return self.get_embeddings_cached(provider, model, api_key)
        except Exception as e:
            st.error(f"Error initializing embeddings: {e}")
            return None
    
    def load_10k_files(self, data_dir: str = "10k_files") -> List[Document]:
        """Load documents using cached function"""
        return load_10k_files_cached(data_dir)
    
    @st.cache_resource
    def create_vector_store_cached(_self, document_chunks: List[Document], embeddings_provider: str, embeddings_model: str, embeddings_key: str):
        """Create vector store - cached to avoid recreation"""
        with performance_monitor("Vector Store Creation"):
            embeddings = _self.get_embeddings_cached(embeddings_provider, embeddings_model, embeddings_key)
            
            if not embeddings:
                return None
            
            # Create vector store with persistence
            vector_store = Chroma.from_documents(
                documents=document_chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            
            logger.info(f"Vector store created with {len(document_chunks)} chunks")
            return vector_store
    
    def create_vector_store(self, documents: List[Document], embeddings_provider: str = None, force_recreate: bool = False):
        """Create vector store with smart caching"""
        if not documents:
            st.error("No documents to process")
            return None
        
        embeddings_provider = embeddings_provider or self.config['EMBEDDINGS']['provider']
        embeddings_model = self.config['EMBEDDINGS']['model']
        
        # Get API key
        if embeddings_provider == 'gemini':
            embeddings_key = self.config['EMBEDDINGS']['api_key'] or os.getenv('GOOGLE_API_KEY')
        elif embeddings_provider == 'openai':
            embeddings_key = self.config['EMBEDDINGS']['api_key'] or os.getenv('OPENAI_API_KEY')
        else:
            embeddings_key = ""
        
        # Create document chunks
        chunk_size = int(self.config['RAG']['chunk_size'])
        chunk_overlap = int(self.config['RAG']['chunk_overlap'])
        
        document_chunks = create_document_chunks_cached(documents, chunk_size, chunk_overlap)
        
        if not document_chunks:
            st.error("No document chunks created")
            return None
        
        # Store chunks for hybrid retrieval
        self.document_splits = document_chunks
        
        try:
            self.vector_store = self.create_vector_store_cached(
                document_chunks, embeddings_provider, embeddings_model, embeddings_key
            )
            
            if self.vector_store:
                st.success(f"Vector store ready with {len(document_chunks)} chunks")
            
            return self.vector_store
            
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            logger.error(f"Vector store creation error: {e}")
            return None
    
    @st.cache_data(ttl=CACHE_TTL_QUERIES)
    def cached_query(_question: str, _config_hash: str, _vector_store_id: str) -> Optional[Dict]:
        """Cache query results to avoid repeated expensive operations"""
        # This is a placeholder - the actual implementation would be in ask_question
        return None
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask question with performance monitoring and caching"""
        start_time = time.time()
        
        if not self.qa_chain:
            return {"error": "QA chain not initialized"}
        
        try:
            with performance_monitor("Question Processing"):
                # Check for comparison questions
                if is_comparison_question(question):
                    # Use per-company retrieval for comparison questions
                    documents = getattr(self, 'document_splits', [])
                    if documents and self.vector_store:
                        context_docs = retrieve_per_company(self.vector_store, question, documents)
                        # Process with custom context
                        result = self.qa_chain({"question": question, "chat_history": []})
                    else:
                        result = self.qa_chain({"question": question})
                else:
                    result = self.qa_chain({"question": question})
                
                # Extract sources
                sources = []
                if 'source_documents' in result:
                    sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
                
                response_time = time.time() - start_time
                
                return {
                    "answer": result.get('answer', ''),
                    "sources": sources,
                    "performance": {
                        "response_time": response_time,
                        "source_count": len(sources),
                        "answer_length": len(result.get('answer', ''))
                    }
                }
                
        except Exception as e:
            st.error(f"Error processing question: {e}")
            return {"error": str(e)}
    
    def create_qa_chain(self, llm_provider: str = None, system_prompt: str = None):
        """Create QA chain with optimizations"""
        if not self.vector_store:
            st.error("Vector store not initialized")
            return None
        
        llm = self.get_llm(llm_provider)
        if not llm:
            return None
        
        # Use optimized retriever
        use_hybrid = self.config['LIGHT_RAG'].get('use_hybrid_retrieval', 'true').lower() == 'true'
        
        if use_hybrid and hasattr(self, 'document_splits'):
            # Create hybrid retriever with caching
            base_retriever = self._create_hybrid_retriever_cached()
        else:
            base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": int(self.config['RAG']['similarity_top_k'])}
            )
        
        # Default prompt (same as original)
        default_system_prompt = """
You are an AI Financial Analyst focused exclusively on interpreting and comparing 10-K annual reports from Alphabet (Google), Amazon, Microsoft, and Apple.

Your Responsibilities
    • Extract and report only the facts contained in the provided 10-K filings.
    • Compare financial metrics (e.g., revenue, cash flow, margins) and business strategies across the four companies.
    • Identify material risks, opportunities, and key business drivers.
    • Answer questions about market position, competitive landscape, and strategic initiatives.
    • Cite precise numbers and dates from the documents.
    • If a topic or data point is missing, state explicitly: "Not mentioned in the provided documents."

Guiding Principles
    • Context-Only: Do not use or infer any information beyond what's in the supplied 10-Ks.
    • No Guesswork: If you're unsure or a fact isn't specified, say so rather than speculate.
    • Data-Driven Comparisons: Anchor all comparisons to exact figures from the filings.
    • Investor Focus: Emphasize material information relevant to investment decisions.
    • Objective & Analytical: Maintain a neutral, evidence-based tone.

Context: {context}
Question: {question}
Chat History: {chat_history}

Answer:
"""
        
        prompt = PromptTemplate(
            template=system_prompt or default_system_prompt,
            input_variables=["context", "question", "chat_history"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=base_retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False,  # Reduce verbosity for performance
            output_key="answer"
        )
        
        return self.qa_chain
    
    @st.cache_resource
    def _create_hybrid_retriever_cached(_self):
        """Create hybrid retriever with caching"""
        if not _self.vector_store or not hasattr(_self, 'document_splits'):
            return None
        
        with performance_monitor("Hybrid Retriever Creation"):
            # Dense retriever
            dense_retriever = _self.vector_store.as_retriever(
                search_kwargs={"k": int(_self.config['LIGHT_RAG']['rerank_top_k'])}
            )
            
            # Sparse retriever
            sparse_retriever = BM25Retriever.from_documents(_self.document_splits)
            sparse_retriever.k = int(_self.config['LIGHT_RAG']['rerank_top_k'])
            
            # Ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                weights=[
                    float(_self.config['LIGHT_RAG']['dense_weight']),
                    float(_self.config['LIGHT_RAG']['sparse_weight'])
                ]
            )
            
            return ensemble_retriever
    
    def manage_session_memory(self):
        """Manage session state memory to prevent unbounded growth"""
        # Limit chat history
        if 'chat_history' in st.session_state:
            if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
        
        # Limit comparison history
        if 'comparison_history' in st.session_state:
            if len(st.session_state.comparison_history) > MAX_COMPARISON_HISTORY:
                st.session_state.comparison_history = st.session_state.comparison_history[-MAX_COMPARISON_HISTORY:]
        
        # Limit performance metrics
        if 'performance_metrics' in st.session_state:
            if len(st.session_state.performance_metrics) > 100:
                st.session_state.performance_metrics = st.session_state.performance_metrics[-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from collected metrics"""
        if 'performance_metrics' not in st.session_state or not st.session_state.performance_metrics:
            return {}
        
        metrics = st.session_state.performance_metrics
        
        # Calculate averages
        durations = [m['duration'] for m in metrics]
        memory_changes = [m['memory_change'] for m in metrics if m['memory_change'] > 0]
        
        return {
            'total_operations': len(metrics),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'avg_memory_change': sum(memory_changes) / len(memory_changes) if memory_changes else 0,
            'operations_by_type': {op: len([m for m in metrics if m['operation'] == op]) 
                                 for op in set(m['operation'] for m in metrics)}
        }

# Main interface function (optimized version of the original main function)
def main():
    st.set_page_config(
        page_title="Optimized 10-K RAG Chatbot",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Optimized 10-K RAG Financial Analysis Chatbot")
    st.write("**Enhanced Performance Version** - Analyzing 10-K reports from Alphabet, Amazon, Microsoft, and Apple")
    
    # Load environment variables
    load_environment_from_config()
    
    # Initialize optimized chatbot
    if 'chatbot_optimized' not in st.session_state:
        with st.spinner("Initializing optimized chatbot..."):
            st.session_state.chatbot_optimized = RAGChatbotOptimized()
    
    chatbot = st.session_state.chatbot_optimized
    
    # Manage memory
    chatbot.manage_session_memory()
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 Optimized Chat Interface")
        
        # Performance metrics display
        perf_summary = chatbot.get_performance_summary()
        if perf_summary:
            with st.expander("⚡ Performance Metrics", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Operations", perf_summary['total_operations'])
                    st.metric("Avg Duration", f"{perf_summary['avg_duration']:.2f}s")
                with col_b:
                    st.metric("Max Duration", f"{perf_summary['max_duration']:.2f}s")
                    st.metric("Avg Memory Change", f"{perf_summary['avg_memory_change']:.1f}MB")
                with col_c:
                    for op_type, count in perf_summary['operations_by_type'].items():
                        st.write(f"**{op_type}**: {count}")
        
        # System initialization section
        st.subheader("⚙️ System Initialization")
        
        init_col1, init_col2 = st.columns(2)
        
        with init_col1:
            if st.button("🔄 Initialize System", help="Load documents and create vector store"):
                with st.spinner("Initializing optimized system..."):
                    # Load documents
                    documents = chatbot.load_10k_files()
                    
                    if documents:
                        st.success(f"✅ Loaded {len(documents)} documents")
                        
                        # Create vector store
                        vector_store = chatbot.create_vector_store(documents)
                        
                        if vector_store:
                            # Create QA chain
                            qa_chain = chatbot.create_qa_chain()
                            
                            if qa_chain:
                                st.success("🎉 Optimized system initialized successfully!")
                            else:
                                st.error("Failed to create QA chain")
                        else:
                            st.error("Failed to create vector store")
                    else:
                        st.error("Failed to load documents")
        
        with init_col2:
            # Clear cache button
            if st.button("🗑️ Clear Cache", help="Clear all cached data"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")
                st.rerun()
        
        # Chat interface
        if chatbot.qa_chain:
            st.subheader("💬 Ask Your Questions")
            
            user_question = st.text_area(
                "Enter your question about the 10-K reports:",
                placeholder="e.g., How much cash does Amazon have at the end of 2024?",
                height=100
            )
            
            if st.button("🚀 Ask Question") and user_question:
                with st.spinner("Processing your question..."):
                    result = chatbot.ask_question(user_question)
                    
                    if 'error' not in result:
                        st.subheader("🤖 Answer")
                        st.write(result['answer'])
                        
                        # Performance metrics
                        if 'performance' in result:
                            perf = result['performance']
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("⏱️ Response Time", f"{perf['response_time']:.2f}s")
                            with col_b:
                                st.metric("📚 Sources Used", perf['source_count'])
                            with col_c:
                                st.metric("📝 Answer Length", f"{perf['answer_length']} chars")
                        
                        # Sources
                        if result['sources']:
                            st.subheader("📚 Sources")
                            for source in result['sources']:
                                st.write(f"- {os.path.basename(source)}")
                        
                        # Store in chat history (with size management)
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': result['answer'],
                            'sources': result['sources'],
                            'performance': result.get('performance', {}),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Auto-manage memory
                        chatbot.manage_session_memory()
                    
                    else:
                        st.error(f"Error: {result['error']}")
        else:
            st.info("Please initialize the system first to start chatting.")
    
    with col2:
        st.header("📊 System Status")
        
        # System status
        status_items = [
            ("Vector Store", "✅ Ready" if chatbot.vector_store else "❌ Not Ready"),
            ("QA Chain", "✅ Ready" if chatbot.qa_chain else "❌ Not Ready"),
            ("Documents Loaded", f"{len(getattr(chatbot, 'document_splits', []))} chunks"),
            ("Cache Status", "✅ Active"),
            ("Memory Management", "✅ Optimized"),
        ]
        
        for item, status in status_items:
            st.write(f"**{item}**: {status}")
        
        # Memory usage info
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        st.metric("Current Memory Usage", f"{current_memory:.1f} MB")
        
        # Chat history (limited display)
        st.subheader("💭 Recent Chat History")
        
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.write(f"**Total conversations**: {len(st.session_state.chat_history)}")
            
            # Show last 3 conversations
            for i, chat in enumerate(st.session_state.chat_history[-3:]):
                with st.expander(f"💬 {chat['question'][:40]}...", expanded=False):
                    st.write(f"**Q**: {chat['question']}")
                    st.write(f"**A**: {chat['answer'][:200]}...")
                    
                    if 'performance' in chat:
                        perf = chat['performance']
                        st.write(f"**Time**: {perf.get('response_time', 0):.2f}s")
            
            # Clear history
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No chat history yet.")
        
        # Performance monitoring
        st.subheader("⚡ Performance Monitor")
        
        if 'performance_metrics' in st.session_state and st.session_state.performance_metrics:
            recent_metrics = st.session_state.performance_metrics[-5:]
            
            for metric in recent_metrics:
                st.write(f"**{metric['operation']}**: {metric['duration']:.2f}s")
        
        # Cache management
        st.subheader("🔧 Cache Management")
        
        cache_info = {
            "Document Cache": "Active" if st.cache_data else "Inactive",
            "Model Cache": "Active" if st.cache_resource else "Inactive",
            "Session Objects": len(st.session_state),
        }
        
        for key, value in cache_info.items():
            st.write(f"**{key}**: {value}")

if __name__ == "__main__":
    main()