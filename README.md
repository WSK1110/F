# 10-K Financial Analysis RAG Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) chatbot for analyzing 10-K annual reports from major technology companies (Alphabet/Google, Amazon, Microsoft). This system supports multiple Language Models and embedding models for comparison and evaluation.

## 🎯 Project Overview

This RAG chatbot is designed to:
- Analyze 10-K financial reports from Alphabet, Amazon, and Microsoft
- Answer complex financial and business questions
- Compare companies across various metrics
- Support multiple LLM providers (Gemini, OpenAI, Ollama)
- Provide source citations for all answers
- Evaluate response quality and detect hallucinations

## 🚀 Features

### Multi-Model Support
- **LLM Providers**: Google Gemini, OpenAI GPT, Ollama (local models)
- **Embedding Models**: Gemini, OpenAI, Ollama embeddings
- **Easy switching** between different model combinations

### Advanced RAG System
- **Document Processing**: PDF loading and text chunking
- **Vector Storage**: ChromaDB for efficient similarity search
- **Conversational Memory**: Maintains context across questions
- **Source Citation**: Provides document sources for all answers

### User-Friendly Interface
- **Streamlit Web App**: Modern, responsive interface
- **Real-time Configuration**: Adjust settings without restart
- **Chat History**: Export conversations for analysis
- **Sample Questions**: Pre-built questions for testing

## 📋 Prerequisites

### Required Software
- Python 3.8+
- Miniconda (recommended)
- VSCode (recommended)

### API Keys (Optional)
- Google API Key (for Gemini models)
- OpenAI API Key (for GPT models)
- Ollama (for local models - no API key needed)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AI
```

### 2. Set Up Python Environment
```bash
# Create conda environment
conda create -n rag-chatbot python=3.9
conda activate rag-chatbot

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Ollama (for Local Models)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 4. Download Ollama Models
```bash
# Download models for local use
ollama pull llama3.1
ollama pull mistral
ollama pull deepseek
ollama pull nomic-embed-text
```

### 5. Set Up API Keys
Create a `.env` file in the project root:
```bash
# Google API Key (for Gemini)
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI API Key (for GPT)
OPENAI_API_KEY=your_openai_api_key_here
```

## 📁 Data Setup

### 1. Create Data Directory
```bash
mkdir 10k_files
```

### 2. Add 10-K PDF Files
Place your 10-K PDF files in the `10k_files` directory:
```
10k_files/
├── alphabet_10k_2024.pdf
├── amazon_10k_2024.pdf
└── microsoft_10k_2024.pdf
```

### 3. File Naming Convention
Use descriptive names for your PDF files to help with source identification.

## 🚀 Usage

### 1. Start the Application
```bash
streamlit run RAG.py
```

### 2. Initialize the System
1. Open the web interface (usually at http://localhost:8501)
2. Click "🔄 Initialize System" to load documents and create the vector store
3. Wait for the system to process your 10-K files

### 3. Configure Models
Use the sidebar to:
- Select LLM provider (Gemini, OpenAI, Ollama)
- Choose specific models
- Adjust RAG settings (chunk size, overlap, top-k)

### 4. Ask Questions
- Use the sample questions provided
- Type your own questions
- View answers with source citations

## 📊 Sample Questions

### Risk Evaluation
- "Do these companies worry about the challenges or business risks in China or India in terms of cloud service?"

### Financial Analysis
- "How much CASH does Amazon have at the end of 2024?"
- "Compared to 2023, does Amazon's liquidity decrease or increase?"

### Business Analysis
- "What is the business where main revenue comes from for Amazon / Google / Microsoft?"
- "What main businesses does Amazon do?"

## 🔧 Configuration

### Model Combinations to Test

#### High Performance (Cloud Models)
- **LLM**: Gemini Pro + **Embeddings**: Gemini
- **LLM**: GPT-4 + **Embeddings**: OpenAI

#### Cost-Effective (Hybrid)
- **LLM**: Ollama (local) + **Embeddings**: Gemini
- **LLM**: Ollama (local) + **Embeddings**: OpenAI

#### Local Only
- **LLM**: Ollama + **Embeddings**: Ollama

### RAG Settings
- **Chunk Size**: 500-2000 characters (default: 850)
- **Chunk Overlap**: 100-500 characters (default: 300)
- **Top-K Results**: 3-10 documents (default: 8)

## 📈 Evaluation Framework

### Response Quality Metrics
- **Term Overlap**: Percentage of expected terms in response
- **Answer Length**: Comprehensive response length
- **Source Citation**: Presence of document sources
- **Factual Accuracy**: Verification against source documents

### Hallucination Detection
- Compare responses against source documents
- Check for unsupported claims
- Verify numerical accuracy
- Cross-reference across multiple sources

## 🧪 Testing Strategy

### Stage 1: Baseline Testing
1. Test with sample questions using Gemini + Gemini
2. Verify accuracy and completeness
3. Document baseline performance

### Stage 2: Model Comparison
1. Test same questions with different model combinations
2. Compare response quality, speed, and cost
3. Identify strengths and weaknesses of each approach

### Stage 3: Boundary Testing
1. Ask complex, multi-part questions
2. Test edge cases and ambiguous queries
3. Explore hallucination scenarios
4. Document limitations and failure modes

## 📝 Project Deliverables

### Code Repository
- Complete RAG chatbot implementation
- Configuration management
- Evaluation framework
- Documentation and setup guides

### Technical Documentation
- Model comparison results
- Performance analysis
- Hallucination detection findings
- Optimization recommendations

### Presentation Materials
- 8-minute team presentation
- Design choices and rationale
- Results and insights
- Challenges and solutions

## 🔍 Advanced Features

### Custom System Prompts
Modify the system prompt in the `create_qa_chain` method to:
- Adjust the assistant's persona
- Change response style
- Add specific instructions
- Improve accuracy for certain question types

### Evaluation Tools
Use the `evaluate_response` method to:
- Compare expected vs actual answers
- Calculate similarity metrics
- Track performance over time
- Identify improvement areas

### Export and Analysis
- Export chat history as JSON
- Analyze conversation patterns
- Track question-answer pairs
- Generate performance reports

## 🚨 Troubleshooting

### Common Issues

#### API Key Errors
```bash
# Check environment variables
echo $GOOGLE_API_KEY
echo $OPENAI_API_KEY
```

#### Ollama Connection Issues
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve
```

#### Memory Issues
- Reduce chunk size for large documents
- Use smaller embedding models
- Clear vector store cache

#### Performance Issues
- Use cloud embeddings with local LLMs
- Adjust top-k parameter
- Optimize chunk overlap

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [10-K Filing Guide](https://www.sec.gov/fast-answers/answersform10khtm.html)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with API usage terms and data privacy regulations.

## 👥 Team

- **Team Members**: [Add your team members here]
- **Roles**: [Add roles and responsibilities]

---

**Note**: This chatbot is designed for educational and research purposes. Always verify financial information from official sources before making investment decisions. 