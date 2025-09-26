#!/bin/bash
# setup.sh - CFO Copilot Team Setup Script
# Run this script to set up the project for new team members

echo "🚀 CFO Copilot - Team Setup Script"
echo "=================================="

# Check if .env.template exists
if [ ! -f ".env.template" ]; then
    echo "❌ Error: .env.template not found!"
    echo "Make sure you're in the project root directory."
    exit 1
fi

# Create .env from template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📋 Creating .env from template..."
    cp .env.template .env
    echo "✅ .env file created!"
    echo ""
    echo "🔑 IMPORTANT: Edit .env and add your actual API keys:"
    echo "   - Get Google API key: https://makersuite.google.com/app/apikey"
    echo "   - Get SERP API key: https://serpapi.com/ (optional)"
    echo ""
else
    echo "⚠️  .env file already exists, skipping..."
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed!"
else
    echo "⚠️  requirements.txt not found, installing core packages..."
    pip install streamlit pandas matplotlib seaborn numpy python-dotenv langchain langchain-experimental langchain-google-genai openpyxl
fi

echo ""
echo "🎯 Setup Complete! Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: streamlit run app.py"
echo "3. Upload data.xlsx and start asking CFO questions!"
echo ""
echo "📚 Need help? Check the README.md file."

# Optional: Open .env file for editing
read -p "🔧 Open .env file for editing now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v code &> /dev/null; then
        code .env
    elif command -v nano &> /dev/null; then
        nano .env
    elif command -v vim &> /dev/null; then
        vim .env
    else
        echo "Please edit .env manually with your preferred editor"
    fi
fi