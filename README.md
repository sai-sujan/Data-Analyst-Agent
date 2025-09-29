# 💼 CFO Copilot

AI-powered financial analysis assistant for CFOs. Upload Excel/CSV data and ask questions in plain English.

<img width="1704" height="911" alt="Screenshot 2025-09-26 at 2 46 23 PM" src="https://github.com/user-attachments/assets/b9d85062-2795-410a-874a-2948e3d1d7cb" />
<img width="1704" height="911" alt="Screenshot 2025-09-26 at 2 46 38 PM" src="https://github.com/user-attachments/assets/627c49e8-e77a-4d39-b6de-32b9f3e5b876" />
<img width="1704" height="911" alt="Screenshot 2025-09-26 at 2 47 34 PM" src="https://github.com/user-attachments/assets/e50e59a2-965d-4e4c-a25c-11d583afa1f0" />


## 🚀 Quick Start

**Option 1: Automated Setup**
```bash
git clone https://github.com/yourusername/cfo-copilot.git
cd cfo-copilot
chmod +x setup.sh
./setup.sh
```

**Option 2: Manual Setup**
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
