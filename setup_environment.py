#!/usr/bin/env python3
"""
Script to set up environment variables from config.ini
"""

import os
import configparser
from pathlib import Path

def setup_environment():
    """Set up environment variables from config.ini"""
    
    config_file = Path("config.ini")
    if not config_file.exists():
        print("❌ config.ini not found")
        return False
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Set Google API key
    if 'LLM' in config and 'google_api_key' in config['LLM']:
        google_key = config['LLM']['google_api_key'].strip()
        if google_key and google_key != '':
            os.environ['GOOGLE_API_KEY'] = google_key
            print("✅ Google API key set from config.ini")
        else:
            print("⚠️  Google API key is empty in config.ini")
    
    # Set OpenAI API key
    if 'LLM' in config and 'openai_api_key' in config['LLM']:
        openai_key = config['LLM']['openai_api_key'].strip()
        if openai_key and openai_key != '':
            os.environ['OPENAI_API_KEY'] = openai_key
            print("✅ OpenAI API key set from config.ini")
        else:
            print("⚠️  OpenAI API key is empty in config.ini")
    
    return True

def check_environment():
    """Check if environment variables are set"""
    
    print("\n🔍 Environment Check:")
    print("=" * 30)
    
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if google_key:
        print(f"✅ GOOGLE_API_KEY: {google_key[:10]}...")
    else:
        print("❌ GOOGLE_API_KEY: Not set")
    
    if openai_key:
        print(f"✅ OPENAI_API_KEY: {openai_key[:10]}...")
    else:
        print("❌ OPENAI_API_KEY: Not set")
    
    return bool(google_key or openai_key)

if __name__ == "__main__":
    print("🚀 Environment Setup Script")
    print("=" * 40)
    
    # Set up environment
    setup_ok = setup_environment()
    
    if setup_ok:
        # Check environment
        env_ok = check_environment()
        
        if env_ok:
            print("\n🎉 Environment setup completed!")
            print("\n💡 You can now run:")
            print("   streamlit run RAG.py")
            print("   python test_gemini_models.py")
        else:
            print("\n⚠️  Environment setup completed but no API keys found")
            print("   Please add your API keys to config.ini")
    else:
        print("\n❌ Environment setup failed")
    
    print("\n" + "=" * 40) 