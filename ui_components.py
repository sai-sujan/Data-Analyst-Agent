import streamlit as st

def render_sidebar(dataframes_dict):
    """Render the sidebar with data overview"""
    st.header("📋 Data Overview")
    
    # Dataset size summary
    total_rows = sum(len(df) for df in dataframes_dict.values())
    if total_rows > 50000:
        st.error(f"⚠️ Very Large Dataset: {total_rows:,} total rows")
        st.markdown("*Data has been optimized for performance*")
    elif total_rows > 20000:
        st.warning(f"📊 Large Dataset: {total_rows:,} total rows")
        st.markdown("*Using optimized analysis methods*")
    else:
        st.success(f"✅ Dataset Size: {total_rows:,} total rows")
    
    # Individual dataset previews
    for name, df in dataframes_dict.items():
        with st.expander(f"{name.upper()} ({len(df):,} rows)"):
            st.dataframe(df.head(), use_container_width=True)
            
            # Show unique values for key columns
            if name in ['actuals', 'budget']:
                if 'account_category' in df.columns:
                    st.write("**Account Categories:**")
                    categories = df['account_category'].unique()[:10]  # First 10
                    st.write(", ".join(categories))
            
            # Data quality indicators
            if len(df) > 10000:
                st.markdown("*🔄 Large dataset - using optimized analysis*")

def render_sample_questions(dataframes_dict, analyzer):
    """Render sample questions section"""
    st.header("💡 Sample Questions")
    st.markdown("*Click any question to ask it:*")
    
    sample_questions = [
        "What was June 2025 revenue vs budget?",
        "Show revenue trends by month for 2024", 
        "Break down operating expenses by category",
        "What is our cash runway based on current burn rate?", 
        "Compare ParentCo vs SubsidiaryA performance",
        "Calculate gross margin trends over time"
    ]
    
    for i, question in enumerate(sample_questions):
        if st.button(question, key=f"sample_btn_{i}", use_container_width=True):
            # Add to chat history and trigger analysis
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Analyze the question
            reply = analyzer.analyze_query(question, dataframes_dict)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            
            st.rerun()

def render_data_metrics(dataframes_dict):
    """Render quick data metrics if available"""
    st.header("📊 Quick Metrics")
    
    try:
        actuals = dataframes_dict.get('actuals')
        if actuals is not None and 'amount' in actuals.columns:
            # Latest month revenue
            revenue_data = actuals[actuals['account_category'].str.contains('Revenue', case=False, na=False)]
            if not revenue_data.empty:
                latest_month = revenue_data['month'].max()
                latest_revenue = revenue_data[revenue_data['month'] == latest_month]['amount'].sum()
                st.metric("Latest Month Revenue", f"${latest_revenue:,.0f}")
            
            # Total expenses
            expense_data = actuals[actuals['account_category'].str.contains('Opex|Expense', case=False, na=False)]
            if not expense_data.empty:
                latest_expenses = expense_data[expense_data['month'] == latest_month]['amount'].sum()
                st.metric("Latest Month Expenses", f"${latest_expenses:,.0f}")
        
        # Cash position
        cash_data = dataframes_dict.get('cash')
        if cash_data is not None and 'cash_usd' in cash_data.columns:
            current_cash = cash_data['cash_usd'].iloc[-1]
            st.metric("Current Cash", f"${current_cash:,.0f}")
            
    except Exception:
        st.info("Upload complete financial data to see metrics")

def render_help_section():
    """Render help and tips section"""
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        **Data Format Tips:**
        - Ensure 'month' column uses YYYY-MM-DD format
        - Use consistent naming: 'Revenue', 'COGS', 'Opex: Category'
        - Include 'entity' column for multi-company analysis
        
        **Best Questions to Ask:**
        - "What was [Month Year] revenue vs budget?"
        - "Show [metric] trends over time"
        - "Break down [category] by [dimension]"
        - "Calculate [financial ratio]"
        
        **Chart Keywords:**
        - Use "show", "chart", "plot", "trend" for visualizations
        - Specify time periods: "last 6 months", "2024", "Q2"
        
        **CFO Metrics Supported:**
        - Revenue vs Budget variance
        - Cash runway and burn rate
        - Gross margin analysis
        - Operating expense breakdown
        - Entity performance comparison
        """)

def render_export_options():
    """Render export options for reports"""
    with st.expander("📤 Export Options"):
        if st.button("📊 Generate Executive Summary"):
            if 'dataframes' in st.session_state and 'analyzer' in st.session_state:
                # Generate summary report
                summary_queries = [
                    "What was the latest month revenue vs budget?",
                    "What is our current cash runway?",
                    "Calculate gross margin for latest month"
                ]
                
                summary_report = "# CFO Executive Summary\n\n"
                
                for query in summary_queries:
                    try:
                        result = st.session_state.analyzer.analyze_query(query, st.session_state.dataframes)
                        summary_report += f"## {query}\n{result}\n\n"
                    except Exception as e:
                        summary_report += f"## {query}\nError: {str(e)}\n\n"
                
                st.download_button(
                    label="📄 Download Summary Report",
                    data=summary_report,
                    file_name="cfo_executive_summary.md",
                    mime="text/markdown"
                )
            else:
                st.warning("Upload data first to generate summary")

def display_error_message(error_msg):
    """Display formatted error message"""
    st.error(f"❌ {error_msg}")
    
def display_success_message(success_msg):
    """Display formatted success message"""
    st.success(f"✅ {success_msg}")

def display_info_message(info_msg):
    """Display formatted info message"""
    st.info(f"ℹ️ {info_msg}")