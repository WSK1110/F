@echo off
REM Startup script for RAG Chatbot (Windows)

echo 🚀 Starting RAG Chatbot...

REM Check if .env file exists
if not exist ".env" (
    echo ❌ .env file not found. Creating one...
    python setup_api_keys.py
)

REM Load environment variables and start Streamlit
echo 📊 Launching Streamlit application...
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if not "%%a"=="#" (
        set %%a=%%b
    )
)

streamlit run RAG.py
pause
