#!/usr/bin/env python3
"""
Setup script for 10-K RAG Chatbot
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install required Python packages"""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def create_directories():
    """Create necessary directories"""
    directories = ["10k_files", "chroma_db", "logs"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def create_env_file():
    """Create .env file template"""
    env_content = """# API Keys for RAG Chatbot
# Add your API keys here (optional - some models work without them)

# Google API Key (for Gemini models)
GOOGLE_API_KEY=your_google_api_key_here

# OpenAI API Key (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Note: Ollama models don't require API keys
"""
    
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("📝 Please add your API keys to the .env file if needed")
    else:
        print("📁 .env file already exists")

def check_ollama():
    """Check if Ollama is installed and running"""
    print("🔍 Checking Ollama installation...")
    
    # Check if ollama command exists
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            
            # Check if ollama is running
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Ollama is running")
                    return True
                else:
                    print("⚠️  Ollama is installed but not running")
                    print("💡 Start Ollama with: ollama serve")
                    return False
            except Exception:
                print("⚠️  Could not check if Ollama is running")
                return False
        else:
            print("❌ Ollama is not properly installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        print("💡 Install Ollama from: https://ollama.ai/download")
        return False

def download_ollama_models():
    """Download recommended Ollama models"""
    models = ["llama3.1", "mistral", "deepseek", "nomic-embed-text"]
    
    print("📥 Downloading Ollama models...")
    for model in models:
        print(f"🔄 Downloading {model}...")
        if run_command(f"ollama pull {model}", f"Downloading {model}"):
            print(f"✅ {model} downloaded successfully")
        else:
            print(f"⚠️  Failed to download {model}")

def main():
    """Main setup function"""
    print("🚀 Setting up 10-K RAG Chatbot")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Check Ollama
    ollama_available = check_ollama()
    
    if ollama_available:
        # Ask if user wants to download models
        response = input("\n📥 Do you want to download recommended Ollama models? (y/n): ")
        if response.lower() in ['y', 'yes']:
            download_ollama_models()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Add your 10-K PDF files to the '10k_files' directory")
    print("2. Add API keys to .env file if using cloud models")
    print("3. Run the chatbot: streamlit run RAG.py")
    print("4. Run evaluation: python evaluation.py")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 