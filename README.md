# 🧠 Data Analyst Agent - ReAct Reasoning

Upload a CSV, ask questions, request charts, and let the AI agent **analyze, reason, and visualize** your data.  
Built using **LangChain**, **Google Gemini**, and **Streamlit**.

---

## 🎯 Key Features

- 📁 Upload and analyze **any CSV** file up to 200MB
- 🤖 Chat with your data using **natural language**
- 📊 Generate **dynamic visualizations** using Gemini + matplotlib/seaborn
- 🧠 Supports **ReAct-based reasoning trace**
- 💡 Leverages tools: `Python`, `Wikipedia`, `SerpAPI`, `LangChain Pandas Agent`

---

## 🖼 Interface Preview


  <div align="center">
  <img width="1614" alt="1" src="https://github.com/user-attachments/assets/7582abad-e086-4b92-9c38-9a2a44622f97" />
  <br/>
  <img width="1614" alt="2" src="https://github.com/user-attachments/assets/956b9108-d865-4ef2-a874-324779b2b416" />
  <br/>
  <img width="1614" alt="3" src="https://github.com/user-attachments/assets/fa48313d-9599-4f38-b326-b9a175857bd1" width="90%"/>
  <br/>
  <img width="1614" alt="4" src="https://github.com/user-attachments/assets/2077a444-9181-4966-9260-b6feb69daf02" />

  </div>

---

## ⚙️ Tech Stack

- `LangChain` + `GoogleGenerativeAI`
- `Streamlit`
- `Matplotlib`, `Seaborn`, `Pandas`
- `langchain_experimental`
- `Wikipedia` and `SerpAPI` tools
- Custom visualization prompt generator

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/data-analyst-agent.git
cd data-analyst-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API Keys (Gemini + SerpAPI)
# Create a .env file or edit directly in app.py

# 4. Run the app
streamlit run app.py
