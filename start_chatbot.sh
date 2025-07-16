#!/bin/bash
# Startup script for RAG Chatbot

echo "🚀 Starting RAG Chatbot..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Creating one..."
    python setup_api_keys.py
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Start Streamlit
echo "📊 Launching Streamlit application..."
streamlit run RAG.py
