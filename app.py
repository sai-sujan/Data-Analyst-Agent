import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

# Import our modules
from data_processor import process_excel_file, preprocess_large_datasets
from financial_analyzer import FinancialAnalyzer
from chart_generator import generate_chart_code, safe_exec_chart_code, is_graph_query
from ui_components import render_sidebar, render_sample_questions

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("🔑 GOOGLE_API_KEY not found! Please check your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Streamlit setup
st.set_page_config(page_title="CFO Copilot", layout="wide")
st.title("💼 CFO Copilot")
st.markdown("Upload your financial Excel file and ask questions about revenue, expenses, margins, and cash flow!")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_file = st.file_uploader(
    "📊 Upload your financial data file", 
    type=["xlsx", "csv"],
    help="Upload Excel file with multiple sheets (actuals, budget, fx, cash) or individual CSV files"
)

if uploaded_file:
    try:
        # Process file
        if uploaded_file.name.endswith('.xlsx'):
            dataframes, file_info = process_excel_file(uploaded_file)
            st.success("✅ Excel file with multiple sheets loaded successfully!")
        else:
            # CSV processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            import pandas as pd
            df = pd.read_csv(tmp_path)
            file_name = uploaded_file.name.lower().replace('.csv', '')
            dataframes = {file_name: df}
            file_info = [f"📋 {file_name}: {len(df)} records"]
            st.success("✅ CSV file loaded successfully!")
        
        st.markdown("**Loaded datasets:** " + " | ".join(file_info))
        
        # Preprocess large datasets
        if any(len(df) > 10000 for df in dataframes.values()):
            with st.spinner("🔄 Optimizing large datasets for better performance..."):
                dataframes = preprocess_large_datasets(dataframes)
        
        # Initialize analyzer
        analyzer = FinancialAnalyzer()
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        user_input = st.chat_input("Ask about revenue vs budget, margins, opex breakdown, cash runway, entity performance...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing financial data..."):
                    reply = analyzer.analyze_query(user_input, dataframes)
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

                # Chart generation
                if is_graph_query(user_input):
                    try:
                        st.markdown("---")
                        st.markdown("### 📊 Executive Chart")
                        chart_code = generate_chart_code(analyzer.llm, dataframes, user_input)
                        
                        with st.expander("📝 Generated Chart Code", expanded=False):
                            st.code(chart_code, language="python")
                        
                        fig = safe_exec_chart_code(chart_code, dataframes)
                        if fig:
                            st.pyplot(fig)
                            st.caption("Executive-ready chart generated for CFO presentation.")
                        else:
                            st.warning("Chart generated but could not be displayed.")
                            
                    except Exception as e:
                        st.error(f"⚠️ Could not generate financial chart: {e}")

        # Sidebar
        with st.sidebar:
            render_sidebar(dataframes)
            render_sample_questions(dataframes, analyzer)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

else:
    # No file uploaded state
    st.info("Please upload your financial Excel file or CSV to begin CFO analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📁 Expected File Structure:
        
        **Excel File (.xlsx) with sheets:**
        - `actuals` - Monthly actual results
        - `budget` - Monthly budget data  
        - `cash` - Cash balances
        - `fx` - Exchange rates
        """)
        
        st.markdown("""
        ### 📊 Data Columns Expected:
        - **Actuals/Budget**: month, entity, account_category, amount, currency
        - **Cash**: month, entity, cash_usd
        - **FX**: month, currency, rate_to_usd
        """)
    
    with col2:
        st.markdown("""
        ### 💼 Sample CFO Questions:
        - "What was June 2025 revenue vs budget?"
        - "Show revenue trends by month for 2024"
        - "Break down operating expenses by category"  
        - "What is our cash runway based on burn rate?"
        - "Compare entity performance across divisions"
        - "Calculate gross margin trends over time"
        """)