# app.py
import os
import streamlit as st
import tempfile
import contextlib
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper

# ========== GRAPH UTILITIES ========== #
def is_graph_query(query: str) -> bool:
    keywords = ["plot", "graph", "chart", "heatmap", "histogram", "scatter", "bar", "line", "distribution", "correlation", "visualize"]
    return any(k in query.lower() for k in keywords)

def generate_chart_code(llm, df, user_query: str) -> str:
    schema = ', '.join(df.columns)
    prompt = f"""
        You are a Python data visualization expert using pandas and matplotlib/seaborn.

        You are given a DataFrame named 'df' with these columns:
        {schema}

        The user asks: "{user_query}"

        Write Python code that generates the correct chart. 
        It must include:
            fig, ax = plt.subplots(figsize=(6, 4))

            ... (your plot code)
            return fig

        ⚠️ Important: Do not include markdown, no ```python. Only return pure Python code.
    """
    code = llm.invoke(prompt).content.strip()
    return code.replace("```python", "").replace("```", "").strip()

def safe_exec_chart_code(code: str, df: pd.DataFrame):
    try:
        code = code.replace("plt.show()", "").strip()
        global_namespace = {"df": df, "plt": plt, "sns": sns, "pd": pd}
        local_namespace = {}
        exec(code, global_namespace, local_namespace)
        return local_namespace.get("fig", None)
    except Exception as e:
        raise RuntimeError(f"❌ Code execution failed: {e}")

# ========== API KEYS ========== #
os.environ["GOOGLE_API_KEY"] = "<api key>"
os.environ["SERPAPI_API_KEY"] = "<api key>"

# ========== Streamlit UI ========== #
st.set_page_config(page_title="🧠 Data Analyst Agent", layout="wide")
st.title("🧠 Data Analyst Agent")
st.markdown("Upload a CSV, then chat with your data — ask questions or request charts!")

# ========== Session State Initialization ========== #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========== File Upload ========== #
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    df = pd.read_csv(tmp_path)
    st.success("✅ File uploaded successfully!")

    # ========== LLM + Agent Setup ========== #
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

    def analyze_csv_query(q):
        csv_agent = create_pandas_dataframe_agent(llm, df, verbose=True, handle_parsing_errors=True, allow_dangerous_code=True)
        return csv_agent.run(q)

    tools = [
        Tool.from_function(PythonREPLTool().run, name="Python", description="Run calculations and Python logic.", allow_dangerous_code=True),
        Tool.from_function(WikipediaAPIWrapper().run, name="Wikipedia", description="Explain statistical concepts."),
        Tool.from_function(SerpAPIWrapper().run, name="WebSearch", description="Look up online data or trends."),
        Tool.from_function(analyze_csv_query, name="CSVAnalyzer", description="Ask questions about the uploaded CSV file.")
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # ========== Chat Display ========== #
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ========== Input Box for Chat ========== #
    user_input = st.chat_input("Ask a question about the data or request a chart...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                trace_output = io.StringIO()
                try:
                    with contextlib.redirect_stdout(trace_output):
                        query = "The dataset is already with you, now read it and " + user_input
                        response = agent.invoke(query)
                    reply = response["output"]
                except Exception as e:
                    reply = f"⚠️ Error: {str(e)}"

                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

                with st.expander("🧠 ReAct Reasoning Trace", expanded=False):
                    st.code(trace_output.getvalue())

            # ========== Chart Output ========== #
            if is_graph_query(user_input):
                try:
                    chart_code = generate_chart_code(llm, df, user_input)
                    st.code(chart_code, language="python")
                    fig = safe_exec_chart_code(chart_code, df)
                    st.pyplot(fig)
                    st.caption("Chart generated from Gemini code based on your prompt.")
                except Exception as e:
                    st.error(f"⚠️ Could not generate chart: {e}")
else:
    st.info("Please upload a CSV file to begin.")
