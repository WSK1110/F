import os
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime
import json
import re

# LLM and Embedding imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# RAG components
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
from langchain.schema import Document

# Configuration
import configparser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment_from_config():
    """Load environment variables from config.ini"""
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        
        # Set Google API key
        if 'LLM' in config and 'google_api_key' in config['LLM']:
            google_key = config['LLM']['google_api_key'].strip()
            if google_key and google_key != '':
                os.environ['GOOGLE_API_KEY'] = google_key
                logger.info("Google API key loaded from config.ini")
        
        # Set OpenAI API key
        if 'LLM' in config and 'openai_api_key' in config['LLM']:
            openai_key = config['LLM']['openai_api_key'].strip()
            if openai_key and openai_key != '':
                os.environ['OPENAI_API_KEY'] = openai_key
                logger.info("OpenAI API key loaded from config.ini")
                
    except Exception as e:
        logger.warning(f"Could not load environment from config: {e}")

# Load environment variables at module import
load_environment_from_config()

def extract_company_name(filename):
    # Example: "Amazon_2023_10K.pdf" -> "Amazon"
    match = re.match(r"([A-Za-z]+)", filename)
    return match.group(1) if match else "Unknown"

def get_all_companies(documents):
    return sorted(set(doc.metadata.get('company', 'Unknown') for doc in documents))

def retrieve_per_company(vector_store, question, documents, k_per_company=2):
    companies = get_all_companies(documents)
    all_chunks = []
    for company in companies:
        company_chunks = vector_store.similarity_search(
            question, 
            k=k_per_company, 
            filter={'company': company}
        )
        if not company_chunks:
            # Add a placeholder chunk if nothing found
            company_chunks = [Document(
                page_content=f"Not mentioned in the provided documents for {company}.",
                metadata={'company': company}
            )]
        all_chunks.extend(company_chunks)
    return all_chunks

def is_comparison_question(question):
    # Simple heuristic, can be improved
    return any(word in question.lower() for word in ['compare', 'versus', 'difference', 'both', 'all companies'])

# Add cached loader right after imports
@st.cache_data(show_spinner=False)
def _cached_load_10k_docs(data_dir: str = "10k_files") -> List[Document]:
    """Load and cache 10-K PDF files.
    This prevents expensive disk I/O and PDF parsing on every Streamlit rerun.
    """
    if not os.path.exists(data_dir):
        return []

    # Local import to avoid heavy dependency cost until actually needed
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

class RAGChatbot:
    """
    A comprehensive RAG chatbot for analyzing 10-K files from major tech companies.
    Supports multiple LLMs and embedding models for comparison and evaluation.
    """
    
    def __init__(self, config_file: str = "config.ini"):
        self.config = self._load_config(config_file)
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # ✅ This tells memory to only store the answer
        )
        self.chat_history = []
        
    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config.read(config_file)

        # Ensure all required sections and defaults exist
        if 'DEFAULT' not in config:
            config['DEFAULT'] = {}
        config['DEFAULT'].setdefault('chunk_size', '850')
        config['DEFAULT'].setdefault('chunk_overlap', '300')
        config['DEFAULT'].setdefault('temperature', '0.1')
        config['DEFAULT'].setdefault('max_tokens', '4000')

        if 'LLM' not in config:
            config['LLM'] = {}
        config['LLM'].setdefault('provider', 'gemini')
        config['LLM'].setdefault('model', 'models/gemini-1.5-pro')
        config['LLM'].setdefault('api_key', '')

        if 'EMBEDDINGS' not in config:
            config['EMBEDDINGS'] = {}
        config['EMBEDDINGS'].setdefault('provider', 'gemini')
        config['EMBEDDINGS'].setdefault('model', 'models/embedding-001')
        config['EMBEDDINGS'].setdefault('api_key', '')

        if 'RAG' not in config:
            config['RAG'] = {}
        config['RAG'].setdefault('chunk_size', '850')
        config['RAG'].setdefault('chunk_overlap', '300')
        config['RAG'].setdefault('similarity_top_k', '8')

        if 'LIGHT_RAG' not in config:
            config['LIGHT_RAG'] = {}
        config['LIGHT_RAG'].setdefault('use_hybrid_retrieval', 'true')
        config['LIGHT_RAG'].setdefault('use_reranking', 'true')
        config['LIGHT_RAG'].setdefault('use_compression', 'true')
        config['LIGHT_RAG'].setdefault('sparse_weight', '0.50')
        config['LIGHT_RAG'].setdefault('dense_weight', '0.80')
        config['LIGHT_RAG'].setdefault('rerank_top_k', '12')
        config['LIGHT_RAG'].setdefault('final_top_k', '3')

        # Save updated config back if any defaults were missing
        with open(config_file, 'w') as f:
            config.write(f)

        return config
    
    def get_llm(self, provider: str = None, model: str = None):
        """Initialize LLM based on provider"""
        provider = provider or self.config['LLM']['provider']
        model = model or self.config['LLM']['model']
        
        if provider == 'gemini':
            api_key = self.config['LLM']['api_key'] or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                st.error("Google API key not found. Please set GOOGLE_API_KEY environment variable or add to config.")
                return None
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=float(self.config['DEFAULT']['temperature']),
                max_output_tokens=int(self.config['DEFAULT']['max_tokens'])
            )
        
        elif provider == 'openai':
            api_key = self.config['LLM']['api_key'] or os.getenv('OPENAI_API_KEY')
            if not api_key:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add to config.")
                return None
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=float(self.config['DEFAULT']['temperature']),
                max_tokens=int(self.config['DEFAULT']['max_tokens'])
            )
        
        elif provider == 'ollama':
            return Ollama(
                model=model,
                temperature=float(self.config['DEFAULT']['temperature'])
            )
        
        else:
            st.error(f"Unsupported LLM provider: {provider}")
            return None
    
    def get_embeddings(self, provider: str = None, model: str = None):
        """Initialize embeddings based on provider"""
        provider = provider or self.config['EMBEDDINGS']['provider']
        model = model or self.config['EMBEDDINGS']['model']
        
        if provider == 'gemini':
            api_key = self.config['EMBEDDINGS']['api_key'] or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                st.error("Google API key not found for embeddings.")
                return None
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=api_key
            )
        
        elif provider == 'openai':
            api_key = self.config['EMBEDDINGS']['api_key'] or os.getenv('OPENAI_API_KEY')
            if not api_key:
                st.error("OpenAI API key not found for embeddings.")
                return None
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key
            )
        
        elif provider == 'ollama':
            return OllamaEmbeddings(model=model)
        
        else:
            st.error(f"Unsupported embedding provider: {provider}")
            return None
    
    def load_10k_files(self, data_dir: str = "10k_files") -> List[Document]:
        """Load 10-K PDF files from directory, using the cached helper to reduce load time."""
        documents = _cached_load_10k_docs(data_dir)

        if not documents:
            st.warning(f"No documents could be loaded from {data_dir}.")
            return []

        # Tag each document with company name (done once thanks to caching)
        for doc in documents:
            filename = doc.metadata.get('source', '')
            company = extract_company_name(filename)
            doc.metadata['company'] = company

        # Display summary (light-weight, avoid re-parsing full text)
        st.success(f"✅ Loaded {len(documents)} 10-K documents from cache")
        st.info(f"Total characters across docs: {sum(len(d.page_content) for d in documents):,}")

        return documents
    
    def create_vector_store(self, documents: List[Document], embeddings_provider: str = None, force_recreate: bool = False):
        """Create vector store from documents with LightRAG optimizations"""
        if not documents:
            st.error("No documents to process")
            return None
        
        embeddings = self.get_embeddings(embeddings_provider)
        if not embeddings:
            return None
        
        # Check embedding dimensions and handle database conflicts
        try:
            # Test embedding dimension
            test_embedding = embeddings.embed_query("test")
            embedding_dim = len(test_embedding)
            st.info(f"Embedding dimension: {embedding_dim}")
            
            # Check if existing database has different dimensions
            if os.path.exists("./chroma_db") and not force_recreate:
                try:
                    # Try to load existing database
                    existing_db = Chroma(
                        persist_directory="./chroma_db",
                        embedding_function=embeddings
                    )
                    # If successful, use existing database
                    self.vector_store = existing_db
                    st.success("Using existing vector store")
                    return self.vector_store
                except Exception as e:
                    if "dimension" in str(e).lower():
                        st.warning("⚠️ Dimension mismatch detected! Existing database has different embedding dimensions.")
                        if st.button("🗑️ Clear Database and Recreate", key="clear_db"):
                            import shutil
                            shutil.rmtree("./chroma_db", ignore_errors=True)
                            st.success("Database cleared. Recreating...")
                            force_recreate = True
                        else:
                            st.error("Please clear the database or use a compatible embedding model.")
                            return None
                    else:
                        st.error(f"Error loading existing database: {e}")
                        return None
            
        except Exception as e:
            st.error(f"Error testing embeddings: {e}")
            return None
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.config['RAG']['chunk_size']),
            chunk_overlap=int(self.config['RAG']['chunk_overlap']),
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        st.info(f"📊 Created {len(splits)} text chunks")
        
        # Show sample chunks for debugging
        if splits:
            st.write(f"📄 Sample chunk 1: {splits[0].page_content[:200]}...")
            st.write(f"📄 Sample chunk 2: {splits[1].page_content[:200]}...")
            st.write(f"📄 Sample chunk 3: {splits[2].page_content[:200]}...")
        else:
            st.error("❌ No text chunks created from documents!")
            return None
        
        # Create vector store
        try:
            st.info("🔧 Creating vector store with Chroma...")
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            
            # Store splits for hybrid retrieval
            self.document_splits = splits
            
            st.success(f"✅ Vector store created successfully with {embedding_dim}-dimensional embeddings")
            st.info(f"📊 Vector store contains {len(splits)} document chunks")
            return self.vector_store
            
        except Exception as e:
            st.error(f"❌ Error creating vector store: {e}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def create_hybrid_retriever(self):
        """Create hybrid retriever combining dense and sparse retrieval"""
        if not self.vector_store or not hasattr(self, 'document_splits'):
            st.error("Vector store not initialized")
            return None
        
        # Dense retriever (vector search)
        dense_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": int(self.config['LIGHT_RAG']['rerank_top_k'])}
        )
        
        # Sparse retriever (BM25)
        sparse_retriever = BM25Retriever.from_documents(self.document_splits)
        sparse_retriever.k = int(self.config['LIGHT_RAG']['rerank_top_k'])
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[
                float(self.config['LIGHT_RAG']['dense_weight']),
                float(self.config['LIGHT_RAG']['sparse_weight'])
            ]
        )
        
        return ensemble_retriever
    
    def create_compression_retriever(self, base_retriever):
        """Create compression retriever for better context selection"""
        if not base_retriever:
            return None
        
        # Create embeddings filter for compression
        embeddings = self.get_embeddings()
        if not embeddings:
            return base_retriever
        
        embeddings_filter = EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=0.7
        )
        
        # Create contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=base_retriever
        )
        
        return compression_retriever
    
    def create_qa_chain(self, llm_provider: str = None, system_prompt: str = None):
        """Create QA chain with LightRAG optimizations"""
        if not self.vector_store:
            st.error("Vector store not initialized. Please load documents first.")
            return None
        
        llm = self.get_llm(llm_provider)
        if not llm:
            return None
        
        # Choose retriever based on LightRAG configuration
        use_hybrid = self.config['LIGHT_RAG'].get('use_hybrid_retrieval', 'true').lower() == 'true'
        use_compression = self.config['LIGHT_RAG'].get('use_compression', 'true').lower() == 'true'
        
        if use_hybrid:
            # Create hybrid retriever
            base_retriever = self.create_hybrid_retriever()
            if not base_retriever:
                st.warning("Failed to create hybrid retriever, falling back to dense retrieval")
                base_retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": int(self.config['LIGHT_RAG']['rerank_top_k'])}
                )
        else:
            # Use traditional dense retrieval
            base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": int(self.config['RAG']['similarity_top_k'])}
            )
        
        # Apply compression if enabled
        if use_compression:
            final_retriever = self.create_compression_retriever(base_retriever)
            if not final_retriever:
                final_retriever = base_retriever
        else:
            final_retriever = base_retriever
        
        # Default system prompt for 10-K analysis
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
            retriever=final_retriever,
            memory=self.memory,  # ✅ Memory is connected here
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True,
            output_key="answer"
        )
        
        # Store retriever type for UI display
        self.retriever_type = "Hybrid + Compression" if (use_hybrid and use_compression) else \
                             "Hybrid" if use_hybrid else \
                             "Dense + Compression" if use_compression else "Dense"
        
        st.success(f"QA chain created successfully with {self.retriever_type} retrieval")
        return self.qa_chain

    def create_independent_qa_chain(self, llm_provider: str = None, system_prompt: str = None):
        """Create QA chain WITHOUT memory for independent model comparison"""
        if not self.vector_store:
            st.error("Vector store not initialized. Please load documents first.")
            return None
        
        # Get proper model configuration for the provider
        if llm_provider:
            model_config = self.get_model_config(llm_provider)
            st.info(f"Using {llm_provider} with model: {model_config['model']}")
            llm = self.get_llm(model_config['provider'], model_config['model'])
        else:
            st.warning("No provider specified, using default configuration")
            llm = self.get_llm()
            
        if not llm:
            return None
        
        # Choose retriever based on LightRAG configuration
        use_hybrid = self.config['LIGHT_RAG'].get('use_hybrid_retrieval', 'true').lower() == 'true'
        use_compression = self.config['LIGHT_RAG'].get('use_compression', 'true').lower() == 'true'
        
        if use_hybrid:
            base_retriever = self.create_hybrid_retriever()
            if not base_retriever:
                st.warning("Failed to create hybrid retriever, falling back to dense retrieval")
                base_retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": int(self.config['LIGHT_RAG']['rerank_top_k'])}
                )
        else:
            base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": int(self.config['RAG']['similarity_top_k'])}
            )
        
        if use_compression:
            final_retriever = self.create_compression_retriever(base_retriever)
            if not final_retriever:
                final_retriever = base_retriever
        else:
            final_retriever = base_retriever
        
        # System prompt for independent analysis (no chat history)
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

Answer:
"""
        
        prompt = PromptTemplate(
            template=system_prompt or default_system_prompt,
            input_variables=["context", "question"]
        )
        
        # Create chain WITHOUT memory for independent responses
        from langchain.chains import RetrievalQA
        
        independent_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=final_retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
        
        return independent_chain

    def ask_question_independent(self, question: str, llm_provider: str = None) -> Dict[str, Any]:
        """Ask a question independently (no conversation memory) for model comparison"""
        import time
        start_time = time.time()
        
        try:
            # Create a fresh chain without memory
            independent_chain = self.create_independent_qa_chain(llm_provider)
            if not independent_chain:
                return None
            
            # Ask the question
            result = independent_chain({"query": question})
            
            # Calculate performance metrics
            response_time = time.time() - start_time
            
            # Extract source documents
            source_docs = result.get('source_documents', [])
            sources = []
            for doc in source_docs:
                if hasattr(doc, 'metadata'):
                    sources.append(doc.metadata.get('source', 'Unknown'))
            
            return {
                'answer': result['result'],
                'sources': list(set(sources)),
                'question': question,
                'performance': {
                    'response_time': response_time,
                    'source_count': len(sources),
                    'answer_length': len(result['result'])
                }
            }
        
        except Exception as e:
            st.error(f"Error processing question independently: {e}")
            st.error(f"Provider: {llm_provider}")
            st.error(f"Error type: {type(e).__name__}")
            return None

    def get_model_config(self, provider: str) -> Dict[str, str]:
        """Get the default model configuration for a provider"""
        model_configs = {
            'gemini': {
                'provider': 'gemini',
                'model': 'models/gemini-1.5-pro'
            },
            'openai': {
                'provider': 'openai', 
                'model': 'gpt-4'
            },
            'ollama': {
                'provider': 'ollama',
                'model': 'NucEniac/DeepSeek:latest'  # Default model
            }
        }
        
        # For Ollama, try to use the selected model from session state
        if provider == 'ollama' and 'selected_ollama_model' in st.session_state:
            model_configs['ollama']['model'] = st.session_state.selected_ollama_model
            
        return model_configs.get(provider, {'provider': provider, 'model': 'unknown'})

    def compare_models(self, question: str, models: List[str]) -> Dict[str, Any]:
        """Compare responses from multiple models for the same question"""
        results = {}
        
        for model in models:
            st.info(f"Testing {model}...")
            result = self.ask_question_independent(question, model)
            if result:
                results[model] = result
        
        return results
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get response with sources and performance metrics"""
        if not self.qa_chain:
            st.error("QA chain not initialized. Please create QA chain first.")
            return None
        
        import time
        start_time = time.time()
        
        try:
            # Always use the default chain interface
            result = self.qa_chain({"question": question})
            
            # Calculate performance metrics
            response_time = time.time() - start_time
            
            # Extract source documents
            source_docs = result.get('source_documents', [])
            sources = []
            for doc in source_docs:
                if hasattr(doc, 'metadata'):
                    sources.append(doc.metadata.get('source', 'Unknown'))
            
            return {
                'answer': result['answer'],
                'sources': list(set(sources)),
                'question': question,
                'performance': {
                    'response_time': response_time,
                    'source_count': len(sources),
                    'answer_length': len(result['answer'])
                }
            }
        
        except Exception as e:
            st.error(f"Error processing question: {e}")
            return None
    
    def evaluate_response(self, question: str, expected_answer: str, actual_answer: str) -> Dict[str, Any]:
        """Evaluate response quality"""
        # Simple evaluation metrics
        evaluation = {
            'question': question,
            'expected': expected_answer,
            'actual': actual_answer,
            'metrics': {}
        }
        
        # Check for key terms from expected answer
        expected_terms = set(expected_answer.lower().split())
        actual_terms = set(actual_answer.lower().split())
        overlap = len(expected_terms.intersection(actual_terms))
        
        evaluation['metrics']['term_overlap'] = overlap / len(expected_terms) if expected_terms else 0
        evaluation['metrics']['answer_length'] = len(actual_answer)
        evaluation['metrics']['has_sources'] = 'sources' in actual_answer.lower()
        
        return evaluation
    
    def load_chat_history(self, file_path: str) -> bool:
        """Load chat history from JSON file"""
        try:
            with open(file_path, 'r') as f:
                history_data = json.load(f)
            return self.load_chat_history_from_data(history_data)
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            return False
    
    def load_chat_history_from_data(self, history_data: List[Dict]) -> bool:
        """Load chat history from data structure"""
        try:
            # Validate the data structure
            if isinstance(history_data, list):
                for item in history_data:
                    if not all(key in item for key in ['question', 'answer', 'timestamp']):
                        st.error("Invalid chat history format")
                        return False
                
                # Store in session state
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.extend(history_data)
                st.success(f"Loaded {len(history_data)} conversations")
                return True
            else:
                st.error("Chat history should contain a list of conversations")
                return False
                
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            return False
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the conversation history"""
        if 'chat_history' not in st.session_state or not st.session_state.chat_history:
            return {}
        
        history = st.session_state.chat_history
        
        # Calculate statistics
        total_questions = len(history)
        total_answer_length = sum(len(chat.get('answer', '')) for chat in history)
        avg_answer_length = total_answer_length / total_questions if total_questions > 0 else 0
        
        # Performance metrics
        response_times = [chat.get('performance', {}).get('response_time', 0) for chat in history]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Source usage
        total_sources = sum(len(chat.get('sources', [])) for chat in history)
        avg_sources = total_sources / total_questions if total_questions > 0 else 0
        
        # Time analysis
        timestamps = [chat.get('timestamp', '') for chat in history]
        if timestamps:
            try:
                from datetime import datetime
                times = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
                session_duration = (times[-1] - times[0]).total_seconds() / 60  # minutes
            except:
                session_duration = 0
        else:
            session_duration = 0
        
        return {
            'total_conversations': total_questions,
            'avg_answer_length': avg_answer_length,
            'avg_response_time': avg_response_time,
            'avg_sources_per_question': avg_sources,
            'session_duration_minutes': session_duration,
            'total_sources_used': total_sources
        }

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="10-K RAG Chatbot",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 10-K Financial Analysis RAG Chatbot")
    st.markdown("Analyze Alphabet, Amazon, and Microsoft 10-K reports with AI")
    
    # Check API keys status
    def check_api_keys():
        google_key = os.getenv('GOOGLE_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        status = {
            'google': bool(google_key),
            'openai': bool(openai_key),
            'google_key': google_key[:10] + "..." if google_key else None,
            'openai_key': openai_key[:10] + "..." if openai_key else None
        }
        return status
    
    api_status = check_api_keys()
    
    # Display API key status
    if not api_status['google'] and not api_status['openai']:
        st.error("❌ No API keys found! Please check your config.ini file or environment variables.")
        st.info("💡 Run 'python setup_environment.py' to set up your environment.")
        st.stop()
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    chatbot = st.session_state.chatbot
    
    # Check if chatbot has the new method, if not recreate it
    if not hasattr(chatbot, 'get_model_config'):
        st.info("🔄 Updating chatbot with new features...")
        st.session_state.chatbot = RAGChatbot()
        chatbot = st.session_state.chatbot
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Status
        st.subheader("🔑 API Keys")
        if api_status['google']:
            st.success(f"✅ Google: {api_status['google_key']}")
        else:
            st.error("❌ Google API key missing")
            
        if api_status['openai']:
            st.success(f"✅ OpenAI: {api_status['openai_key']}")
        else:
            st.warning("⚠️ OpenAI API key missing (optional)")
        
        # LLM Configuration
        st.subheader("Language Model")
        llm_provider = st.selectbox(
            "LLM Provider",
            ["gemini", "openai", "ollama"],
            index=0
        )
        if llm_provider == "openai":
            llm_model = st.selectbox("OpenAI Model", ["gpt-4", "gpt-3.5-turbo"])
        elif llm_provider == "gemini":
            llm_model = st.selectbox("Gemini Model", ["models/gemini-1.5-pro", "models/gemini-1.5-flash", "models/gemini-2.0-flash"])
        else:  # ollama
            llm_model = st.selectbox("Ollama Model", ["NucEniac/DeepSeek:latest", "llama3.1", "mistral", "deepseek"])
            
        # Store the selected model for independent mode
        if 'selected_ollama_model' not in st.session_state:
            st.session_state.selected_ollama_model = llm_model
        
        # Embedding Configuration
        st.subheader("Embeddings")
        embedding_provider = st.selectbox(
            "Embedding Provider",
            ["gemini", "openai", "ollama"],
            index=0
        )
        
        if embedding_provider == "gemini":
            embedding_model = st.selectbox("Gemini Embedding", ["models/embedding-001", "models/text-embedding-004"])
        elif embedding_provider == "openai":
            embedding_model = st.selectbox("OpenAI Embedding", ["text-embedding-ada-002"])
        else:  # ollama
            embedding_model = st.selectbox("Ollama Embedding", ["llama3.1", "nomic-embed-text"])
        
        # RAG Configuration
        st.subheader("RAG Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 850, 100)
        chunk_overlap = st.slider("Chunk Overlap", 100, 500, 300, 50)
        top_k = st.slider("Top K Results", 3, 10, 8)
        
        # LightRAG Configuration
        st.subheader("🚀 LightRAG Optimizations")
        use_hybrid = st.checkbox("Use Hybrid Retrieval", value=True, 
                               help="Combine dense and sparse retrieval for better results")
        use_compression = st.checkbox("Use Context Compression", value=True,
                                    help="Filter irrelevant context to improve quality")
        
        if use_hybrid:
            col1, col2 = st.columns(2)
            with col1:
                dense_weight = st.slider("Dense Weight", 0.1, 0.9, 0.80, 0.1,
                                       help="Weight for vector similarity search")
            with col2:
                sparse_weight = st.slider("Sparse Weight", 0.1, 0.9, 0.50, 0.1,
                                        help="Weight for keyword-based search")
        
        rerank_top_k = st.slider("Rerank Top-K", 5, 20, 12, 1,
                                help="Number of documents to consider before final selection")
        final_top_k = st.slider("Final Top-K", 2, 8, 3, 1,
                               help="Final number of documents to use for generation")
        
        # Update config
        chatbot.config['RAG']['chunk_size'] = str(chunk_size)
        chatbot.config['RAG']['chunk_overlap'] = str(chunk_overlap)
        chatbot.config['RAG']['similarity_top_k'] = str(top_k)
        chatbot.config['LIGHT_RAG']['use_hybrid_retrieval'] = str(use_hybrid).lower()
        chatbot.config['LIGHT_RAG']['use_compression'] = str(use_compression).lower()
        if use_hybrid:
            chatbot.config['LIGHT_RAG']['dense_weight'] = str(dense_weight)
            chatbot.config['LIGHT_RAG']['sparse_weight'] = str(sparse_weight)
        chatbot.config['LIGHT_RAG']['rerank_top_k'] = str(rerank_top_k)
        chatbot.config['LIGHT_RAG']['final_top_k'] = str(final_top_k)
        chatbot.config['LLM']['provider'] = llm_provider
        chatbot.config['LLM']['model'] = llm_model
        chatbot.config['EMBEDDINGS']['provider'] = embedding_provider
        chatbot.config['EMBEDDINGS']['model'] = embedding_model
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Initialize system
        if st.button("🔄 Initialize System"):
            # Clear existing Chroma database
            chroma_path = "./chroma_db"
            if os.path.exists(chroma_path):
                with st.spinner("🗑️ Clearing existing vector database..."):
                    import shutil
                    shutil.rmtree(chroma_path, ignore_errors=True)
                    st.success("✅ Existing database cleared")
            
            with st.spinner("Loading 10-K documents..."):
                documents = chatbot.load_10k_files()
                if documents:
                    with st.spinner("Creating vector store..."):
                        chatbot.create_vector_store(documents, embedding_provider, force_recreate=True)
                    with st.spinner("Creating QA chain..."):
                        chatbot.create_qa_chain(llm_provider)
        
        # Chat mode selection
        chat_mode = st.radio(
            "Chat Mode",
            ["Conversational", "Independent (Model Comparison)"],
            help="Conversational: Maintains chat history. Independent: Each question is answered fresh without context."
        )
        
        # Chat interface
        if chatbot.vector_store:
            # Sample questions
            st.subheader("📋 Sample Questions")
            sample_questions = [
                "Do these companies worry about the challenges or business risks in China or India in terms of cloud service?",
                "How much CASH does Amazon have at the end of 2024?",
                "Compared to 2023, does Amazon's liquidity decrease or increase?",
                "What is the business where main revenue comes from for Amazon / Google / Microsoft?",
                "What main businesses does Amazon do?"
            ]
            
            for i, question in enumerate(sample_questions):
                if st.button(f"Q{i+1}: {question[:50]}...", key=f"sample_{i}"):
                    st.session_state.user_question = question
            
            # User input
            user_question = st.text_input(
                "Ask a question about the 10-K reports:",
                value=st.session_state.get('user_question', ''),
                key="user_input"
            )
            
            if chat_mode == "Conversational":
                # Conversational mode with memory
                if not chatbot.qa_chain:
                    st.info("Please initialize the system first to start chatting.")
                else:
                    if st.button("Ask") and user_question:
                        with st.spinner("Processing your question..."):
                            result = chatbot.ask_question(user_question)
                            
                            if result:
                                st.subheader("🤖 Answer")
                                st.write(result['answer'])
                                
                                # Display performance metrics
                                if 'performance' in result:
                                    perf = result['performance']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Response Time", f"{perf['response_time']:.2f}s")
                                    with col2:
                                        st.metric("Sources Used", perf['source_count'])
                                    with col3:
                                        st.metric("Answer Length", f"{perf['answer_length']} chars")
                                
                                if result['sources']:
                                    st.subheader("📚 Sources")
                                    for source in result['sources']:
                                        st.write(f"- {os.path.basename(source)}")
                                
                                # Store in chat history
                                if 'chat_history' not in st.session_state:
                                    st.session_state.chat_history = []
                                
                                st.session_state.chat_history.append({
                                    'question': user_question,
                                    'answer': result['answer'],
                                    'sources': result['sources'],
                                    'performance': result.get('performance', {}),
                                    'timestamp': datetime.now().isoformat()
                                })
            
            else:
                # Independent mode for model comparison
                st.subheader("🔬 Model Comparison Mode")
                st.info("This mode answers each question independently without conversation memory, perfect for comparing different models.")
                
                # Model selection for comparison
                available_models = []
                model_descriptions = {}
                
                if api_status['google']:
                    available_models.extend(['gemini'])
                    model_descriptions['gemini'] = 'Gemini 1.5 Pro'
                if api_status['openai']:
                    available_models.extend(['openai'])
                    model_descriptions['openai'] = 'GPT-4'
                available_models.extend(['ollama'])  # Always available locally
                model_descriptions['ollama'] = 'DeepSeek (Local)'
                
                # Show available models with descriptions
                st.write("**Available Models:**")
                for model in available_models:
                    st.write(f"• {model.upper()}: {model_descriptions[model]}")
                
                selected_models = st.multiselect(
                    "Select models to compare:",
                    available_models,
                    default=available_models[:2] if len(available_models) >= 2 else available_models,
                    help="Choose which models to compare. Each will answer the same question independently."
                )
                
                if st.button("Compare Models") and user_question and selected_models:
                    st.subheader("📊 Model Comparison Results")
                    
                    # Compare models
                    comparison_results = chatbot.compare_models(user_question, selected_models)
                    
                    if comparison_results:
                        # Display results in tabs
                        tabs = st.tabs([f"🤖 {model.upper()}" for model in comparison_results.keys()])
                        
                        for i, (model, result) in enumerate(comparison_results.items()):
                            with tabs[i]:
                                # Show which specific model was used
                                try:
                                    model_config = chatbot.get_model_config(model)
                                    st.write(f"**Model:** {model_config['model']}")
                                except AttributeError:
                                    # Fallback if method doesn't exist (for old session state)
                                    st.write(f"**Model:** {model.upper()}")
                                st.write(f"**Question:** {result['question']}")
                                st.write(f"**Answer:** {result['answer']}")
                                
                                # Performance metrics
                                if 'performance' in result:
                                    perf = result['performance']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Response Time", f"{perf['response_time']:.2f}s")
                                    with col2:
                                        st.metric("Sources Used", perf['source_count'])
                                    with col3:
                                        st.metric("Answer Length", f"{perf['answer_length']} chars")
                                
                                if result['sources']:
                                    st.write(f"**Sources:** {', '.join([os.path.basename(s) for s in result['sources']])}")
                        
                        # Comparison summary
                        st.subheader("📈 Comparison Summary")
                        summary_data = []
                        for model, result in comparison_results.items():
                            perf = result.get('performance', {})
                            summary_data.append({
                                'Model': model.upper(),
                                'Response Time (s)': f"{perf.get('response_time', 0):.2f}",
                                'Sources Used': perf.get('source_count', 0),
                                'Answer Length': perf.get('answer_length', 0)
                            })
                        
                        if summary_data:
                            import pandas as pd
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                        
                        # Store comparison in history
                        if 'comparison_history' not in st.session_state:
                            st.session_state.comparison_history = []
                        
                        st.session_state.comparison_history.append({
                            'question': user_question,
                            'models': selected_models,
                            'results': comparison_results,
                            'timestamp': datetime.now().isoformat()
                        })
                elif not selected_models:
                    st.warning("Please select at least one model to compare.")
        
        else:
            st.info("Please initialize the system first to start chatting.")
    
    with col2:
        st.header("📊 System Status")
        
        # System status
        status_items = [
            ("Vector Store", "✅ Ready" if chatbot.vector_store else "❌ Not Ready"),
            ("QA Chain", "✅ Ready" if chatbot.qa_chain else "❌ Not Ready"),
            ("Retrieval Type", getattr(chatbot, 'retriever_type', 'Not Set')),
            ("LLM Provider", llm_provider),
            ("Embedding Provider", embedding_provider),
            ("Hybrid Retrieval", "✅ Enabled" if use_hybrid else "❌ Disabled"),
            ("Context Compression", "✅ Enabled" if use_compression else "❌ Disabled"),
        ]
        
        for item, status in status_items:
            st.write(f"{item}: {status}")
        
        # Chat history management
        st.subheader("💭 Chat History")
        
        # Import chat history
        uploaded_file = st.file_uploader("📁 Import Chat History (JSON)", type=['json'])
        if uploaded_file is not None:
            try:
                history_data = json.load(uploaded_file)
                if chatbot.load_chat_history_from_data(history_data):
                    st.success("Chat history imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing chat history: {e}")
        
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            # Chat history summary
            summary = chatbot.get_conversation_summary()
            if summary:
                with st.expander("📊 Conversation Summary", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Questions", summary['total_conversations'])
                        st.metric("Avg Response Time", f"{summary['avg_response_time']:.2f}s")
                        st.metric("Session Duration", f"{summary['session_duration_minutes']:.1f} min")
                    with col2:
                        st.metric("Avg Answer Length", f"{summary['avg_answer_length']:.0f} chars")
                        st.metric("Avg Sources", f"{summary['avg_sources_per_question']:.1f}")
                        st.metric("Total Sources", summary['total_sources_used'])
            
            # Chat history controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Clear History", help="Clear all chat history"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            with col2:
                if st.button("📥 Export JSON", help="Download chat history as JSON"):
                    chat_data = json.dumps(st.session_state.chat_history, indent=2)
                    st.download_button(
                        label="Download",
                        data=chat_data,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("📊 Export CSV", help="Download chat history as CSV"):
                    import pandas as pd
                    df = pd.DataFrame(st.session_state.chat_history)
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # Display chat history
            st.write(f"**Total conversations:** {len(st.session_state.chat_history)}")
            
            # Filter options
            show_all = st.checkbox("Show all conversations", value=False)
            display_count = len(st.session_state.chat_history) if show_all else min(5, len(st.session_state.chat_history))
            
            for i, chat in enumerate(st.session_state.chat_history[-display_count:]):
                with st.expander(f"💬 {chat['question'][:60]}...", expanded=False):
                    st.write(f"**Question:** {chat['question']}")
                    st.write(f"**Answer:** {chat['answer']}")
                    
                    # Show performance metrics if available
                    if 'performance' in chat and chat['performance']:
                        perf = chat['performance']
                        st.write(f"**Response Time:** {perf.get('response_time', 'N/A'):.2f}s")
                        st.write(f"**Sources Used:** {perf.get('source_count', 'N/A')}")
                        st.write(f"**Answer Length:** {perf.get('answer_length', 'N/A')} chars")
                    
                    if chat['sources']:
                        st.write(f"**Sources:** {', '.join([os.path.basename(s) for s in chat['sources']])}")
        
                    st.write(f"**Timestamp:** {chat['timestamp']}")
                    
                    # Reuse question button
                    if st.button(f"🔄 Reuse Question", key=f"reuse_{i}"):
                        st.session_state.user_question = chat['question']
                        st.rerun()
        else:
            st.info("No chat history yet. Start asking questions to build your conversation history!")
        
        # Model comparison history
        st.subheader("🔬 Model Comparisons")
        
        if 'comparison_history' in st.session_state and st.session_state.comparison_history:
            st.write(f"**Total comparisons:** {len(st.session_state.comparison_history)}")
            
            # Display recent comparisons
            for i, comparison in enumerate(st.session_state.comparison_history[-3:]):  # Show last 3
                with st.expander(f"🔬 {comparison['question'][:50]}...", expanded=False):
                    st.write(f"**Question:** {comparison['question']}")
                    st.write(f"**Models compared:** {', '.join(comparison['models'])}")
                    st.write(f"**Timestamp:** {comparison['timestamp']}")
                    
                    # Quick performance comparison
                    if 'results' in comparison:
                        perf_data = []
                        for model, result in comparison['results'].items():
                            perf = result.get('performance', {})
                            perf_data.append({
                                'Model': model.upper(),
                                'Time (s)': f"{perf.get('response_time', 0):.2f}",
                                'Sources': perf.get('source_count', 0),
                                'Length': perf.get('answer_length', 0)
                            })
                        
                        if perf_data:
                            import pandas as pd
                            perf_df = pd.DataFrame(perf_data)
                            st.dataframe(perf_df, use_container_width=True)
                    
                    # Reuse question for comparison
                    if st.button(f"🔄 Reuse for Comparison", key=f"reuse_comp_{i}"):
                        st.session_state.user_question = comparison['question']
                        st.rerun()
            
            # Clear comparison history
            if st.button("🗑️ Clear Comparisons", help="Clear all comparison history"):
                st.session_state.comparison_history = []
                st.rerun()
        else:
            st.info("No model comparisons yet. Use the Independent mode to compare models!")
        
        # Debug section
        with st.expander("🐛 Debug Information", expanded=False):
            st.write("**Environment Variables:**")
            st.write(f"GOOGLE_API_KEY: {'Set' if os.getenv('GOOGLE_API_KEY') else 'Not Set'}")
            st.write(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set'}")
            
            st.write("**Config File:**")
            if os.path.exists('config.ini'):
                st.write("✅ config.ini exists")
                try:
                    config = configparser.ConfigParser()
                    config.read('config.ini')
                    if 'LLM' in config:
                        st.write(f"LLM Provider: {config['LLM'].get('provider', 'Not set')}")
                        st.write(f"LLM Model: {config['LLM'].get('model', 'Not set')}")
                except Exception as e:
                    st.write(f"❌ Error reading config: {e}")
            else:
                st.write("❌ config.ini not found")
            
            st.write("**System Status:**")
            st.write(f"Vector Store: {'Ready' if chatbot.vector_store else 'Not Ready'}")
            st.write(f"QA Chain: {'Ready' if chatbot.qa_chain else 'Not Ready'}")
            st.write(f"Documents Loaded: {len(chatbot.document_splits) if hasattr(chatbot, 'document_splits') else 0}")

if __name__ == "__main__":
    main()
