# 💼 CFO Copilot

AI-powered financial analysis assistant for CFOs. Upload Excel/CSV data and ask questions in plain English.

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/cfo-copilot.git
cd cfo-copilot
pip install -r requirements.txt
cp .env.template .env  # Add your Google API key
streamlit run app.py
```

Get your free API key: [Google AI Studio](https://makersuite.google.com/app/apikey)

## 💡 What You Can Ask

- "What was June 2025 revenue vs budget?"
- "Show me cash runway trends" 
- "Break down operating expenses by category"
- "Calculate gross margin for last 3 months"
- "How long will our cash last?"

## 📊 Demo Data

Try it with the sample files in `fixtures/`:
- `actuals.csv` - Monthly actual results
- `budget.csv` - Budget data
- `cash.csv` - Cash balances

## 🧪 Testing

```bash
python -m pytest tests/
```

## 🛠️ Built With

- **Streamlit** - Web interface
- **Google Gemini** - AI analysis
- **LangChain** - Agent framework
- **Pandas** - Data processing

## 📝 License

MIT License - feel free to use and modify!