import os
import streamlit as st
import tempfile
import contextlib
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not GOOGLE_API_KEY:
    st.error("🔑 GOOGLE_API_KEY not found! Please check your .env file.")
    st.stop()

# Set environment variables for libraries
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if SERPAPI_API_KEY:
    os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
# ========== EXCEL FILE PROCESSING ========== #
def process_excel_file(uploaded_file):
    """Process Excel file with multiple financial sheets"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # Read all sheets from Excel file
    excel_sheets = pd.read_excel(tmp_path, sheet_name=None)  # None = read all sheets
    
    dataframes = {}
    file_info = []
    
    for sheet_name, df in excel_sheets.items():
        sheet_name_lower = sheet_name.lower()
        
        # Map sheet names to our expected keys
        if 'actual' in sheet_name_lower:
            dataframes['actuals'] = df
            file_info.append(f"📈 Actuals: {len(df)} records")
        elif 'budget' in sheet_name_lower:
            dataframes['budget'] = df
            file_info.append(f"📊 Budget: {len(df)} records")
        elif 'cash' in sheet_name_lower:
            dataframes['cash'] = df
            file_info.append(f"💰 Cash: {len(df)} records")
        elif 'fx' in sheet_name_lower or 'exchange' in sheet_name_lower:
            dataframes['fx'] = df
            file_info.append(f"💱 FX Rates: {len(df)} records")
        else:
            # Include other sheets with their original names
            dataframes[sheet_name_lower] = df
            file_info.append(f"📋 {sheet_name}: {len(df)} records")
    
    return dataframes, file_info

# ========== GRAPH DETECTION FOR CFO METRICS ========== #
def is_graph_query(query: str) -> bool:
    """Detect if CFO query requests visualization"""
    keywords = [
        "plot", "graph", "chart", "show", "visualize", "trend", 
        "breakdown", "comparison", "over time", "by month", "dashboard"
    ]
    return any(k in query.lower() for k in keywords)

# ========== SAFE CHART EXECUTION WITH ERROR HANDLING ========== #
def safe_exec_chart_code(code: str, dataframes_dict: dict):
    """Safely execute chart generation code with comprehensive error handling"""
    try:
        # Clean the code
        code = code.replace("plt.show()", "").strip()
        
        # Set up namespace with all required imports and data
        global_namespace = {
            "dataframes": dataframes_dict,
            "plt": plt, 
            "sns": sns, 
            "pd": pd,
            "np": np
        }
        
        # Add individual dataframes for easier access
        for name, df in dataframes_dict.items():
            global_namespace[f"{name}_df"] = df
            
        local_namespace = {}
        
        # Execute the code
        exec(code, global_namespace, local_namespace)
        
        # Return the figure
        fig = local_namespace.get("fig", None)
        if fig is None:
            # Try to get current figure
            fig = plt.gcf()
            
        return fig
        
    except Exception as e:
        # Create an informative error chart
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        error_msg = f'''Chart Generation Error: {str(e)}

Try these requests:
• "Show revenue vs budget"
• "What is our cash runway?"  
• "Break down opex by category"
• "Show margin trends"'''
        
        ax.text(0.5, 0.5, error_msg, ha='center', va='center', transform=ax.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        plt.title('Chart Generation Failed', fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        return fig

# ========== ROBUST CHART GENERATION FOR CFO METRICS ========== #
def generate_chart_code(llm, dataframes_dict, user_query: str) -> str:
    """Generate reliable, tested chart code for CFO metrics"""
    
    # Analyze query to determine chart type
    query_lower = user_query.lower()
    
    # Revenue vs Budget Chart
    if any(word in query_lower for word in ['revenue', 'sales']) and any(word in query_lower for word in ['budget', 'vs', 'actual', 'comparison']):
        return '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    # Get dataframes safely
    actuals_df = dataframes.get('actuals', pd.DataFrame())
    budget_df = dataframes.get('budget', pd.DataFrame())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if not actuals_df.empty and not budget_df.empty and 'account_category' in actuals_df.columns:
        # Filter revenue data
        actuals_rev = actuals_df[actuals_df['account_category'].str.contains('Revenue', case=False, na=False)]
        budget_rev = budget_df[budget_df['account_category'].str.contains('Revenue', case=False, na=False)]
        
        if not actuals_rev.empty and not budget_rev.empty:
            # Convert month to datetime and get recent data
            actuals_rev = actuals_rev.copy()
            budget_rev = budget_rev.copy()
            actuals_rev['month'] = pd.to_datetime(actuals_rev['month'])
            budget_rev['month'] = pd.to_datetime(budget_rev['month'])
            
            # Group by month and sum
            actual_monthly = actuals_rev.groupby('month')['amount'].sum().reset_index()
            budget_monthly = budget_rev.groupby('month')['amount'].sum().reset_index()
            
            # Get last 6 months
            if len(actual_monthly) > 0:
                recent_months = actual_monthly.nlargest(6, 'month')['month'].sort_values()
                actual_recent = actual_monthly[actual_monthly['month'].isin(recent_months)]
                budget_recent = budget_monthly[budget_monthly['month'].isin(recent_months)]
                
                # Merge data
                merged = pd.merge(actual_recent, budget_recent, on='month', suffixes=('_actual', '_budget'), how='outer')
                merged = merged.sort_values('month').fillna(0)
                
                if len(merged) > 0:
                    months = [d.strftime('%Y-%m') for d in merged['month']]
                    x_pos = np.arange(len(months))
                    width = 0.35
                    
                    bars1 = ax.bar(x_pos - width/2, merged['amount_actual'], width, label='Actual', color='#1f77b4', alpha=0.8)
                    bars2 = ax.bar(x_pos + width/2, merged['amount_budget'], width, label='Budget', color='#ff7f0e', alpha=0.8)
                    
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(months, rotation=45)
                    ax.legend()
                    
                    plt.title('Revenue: Actual vs Budget', fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Month', fontsize=14)
                    plt.ylabel('Revenue (USD)', fontsize=14)
                    
                    # Add value labels on bars
                    for bar in bars1:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height, f'${height/1000:.0f}K', 
                                   ha='center', va='bottom', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No recent revenue data found', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            else:
                ax.text(0.5, 0.5, 'No revenue data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No revenue data found in datasets', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    else:
        ax.text(0.5, 0.5, 'Revenue data not available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        plt.title('Revenue Analysis', fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
except Exception as e:
    ax.text(0.5, 0.5, f'Chart Error: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    plt.title('Revenue vs Budget Analysis', fontsize=16, fontweight='bold', pad=20)
'''

    # Cash Runway/Trend Chart  
    elif any(word in query_lower for word in ['cash', 'runway', 'burn']):
        return '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    cash_df = dataframes.get('cash', pd.DataFrame())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if not cash_df.empty and 'cash_usd' in cash_df.columns:
        cash_data = cash_df.copy()
        cash_data['month'] = pd.to_datetime(cash_data['month'])
        cash_data = cash_data.sort_values('month')
        
        # Plot cash balance trend
        ax.plot(cash_data['month'], cash_data['cash_usd'], marker='o', linewidth=3, 
               markersize=8, color='#2ca02c', label='Cash Balance')
        
        # Calculate and show burn rate
        if len(cash_data) >= 3:
            recent_cash = cash_data.tail(3)
            cash_changes = recent_cash['cash_usd'].diff().dropna()
            avg_burn = -cash_changes.mean()
            
            if avg_burn > 0:
                current_cash = cash_data['cash_usd'].iloc[-1]
                runway_months = current_cash / avg_burn
                
                ax.text(0.02, 0.98, f'Current Cash: ${current_cash:,.0f}\\nAvg Monthly Burn: ${avg_burn:,.0f}\\nRunway: {runway_months:.1f} months', 
                       transform=ax.transAxes, fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.title('Cash Balance Trend & Runway Analysis', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Cash Balance (USD)', fontsize=14)
        plt.xticks(rotation=45)
        
        # Format y-axis
        max_val = cash_data['cash_usd'].max()
        if max_val >= 1000000:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.1f}M'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            
    else:
        ax.text(0.5, 0.5, 'No cash data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        plt.title('Cash Analysis', fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
except Exception as e:
    ax.text(0.5, 0.5, f'Chart Error: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    plt.title('Cash Flow Analysis', fontsize=16, fontweight='bold', pad=20)
'''

    # Operating Expenses Breakdown
    elif any(word in query_lower for word in ['opex', 'expense', 'breakdown', 'cost']):
        return '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    actuals_df = dataframes.get('actuals', pd.DataFrame())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if not actuals_df.empty and 'account_category' in actuals_df.columns:
        # Filter for operating expenses
        opex_data = actuals_df[actuals_df['account_category'].str.contains('Opex', case=False, na=False)]
        
        if not opex_data.empty:
            # Get most recent month
            opex_data = opex_data.copy()
            opex_data['month'] = pd.to_datetime(opex_data['month'])
            latest_month = opex_data['month'].max()
            recent_opex = opex_data[opex_data['month'] == latest_month]
            
            # Group by category and sum
            opex_summary = recent_opex.groupby('account_category')['amount'].sum().reset_index()
            opex_summary = opex_summary.sort_values('amount', ascending=True)
            
            if len(opex_summary) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(opex_summary)))
                bars = ax.barh(opex_summary['account_category'], opex_summary['amount'], color=colors)
                
                plt.title(f'Operating Expenses Breakdown - {latest_month.strftime("%Y-%m")}', 
                         fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Amount (USD)', fontsize=14)
                plt.ylabel('Expense Category', fontsize=14)
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    if width > 0:
                        ax.text(width, bar.get_y() + bar.get_height()/2, f'${width/1000:.0f}K', 
                               ha='left', va='center', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No operating expense data found', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            ax.text(0.5, 0.5, 'No Opex data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    else:
        ax.text(0.5, 0.5, 'Expense data not available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        plt.title('Operating Expenses Analysis', fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
except Exception as e:
    ax.text(0.5, 0.5, f'Chart Error: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    plt.title('Expense Breakdown', fontsize=16, fontweight='bold', pad=20)
'''

    # Gross Margin Trend
    elif any(word in query_lower for word in ['margin', 'gross', 'profitability']):
        return '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    actuals_df = dataframes.get('actuals', pd.DataFrame())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if not actuals_df.empty and 'account_category' in actuals_df.columns:
        data = actuals_df.copy()
        data['month'] = pd.to_datetime(data['month'])
        
        # Get revenue and COGS by month
        revenue_data = data[data['account_category'].str.contains('Revenue', case=False, na=False)]
        cogs_data = data[data['account_category'].str.contains('COGS|Cost', case=False, na=False)]
        
        if not revenue_data.empty and not cogs_data.empty:
            revenue_monthly = revenue_data.groupby('month')['amount'].sum().reset_index()
            cogs_monthly = cogs_data.groupby('month')['amount'].sum().reset_index()
            
            # Merge and calculate margin
            margin_data = pd.merge(revenue_monthly, cogs_monthly, on='month', suffixes=('_rev', '_cogs'), how='inner')
            margin_data['gross_margin_pct'] = ((margin_data['amount_rev'] - margin_data['amount_cogs']) / margin_data['amount_rev']) * 100
            margin_data = margin_data.sort_values('month')
            
            # Get last 6 months
            recent_margin = margin_data.tail(6)
            
            if len(recent_margin) > 0:
                ax.plot(recent_margin['month'], recent_margin['gross_margin_pct'], 
                       marker='o', linewidth=3, markersize=8, color='#ff7f0e', label='Gross Margin %')
                
                # Add trend line
                if len(recent_margin) > 1:
                    x_numeric = range(len(recent_margin))
                    z = np.polyfit(x_numeric, recent_margin['gross_margin_pct'], 1)
                    p = np.poly1d(z)
                    ax.plot(recent_margin['month'], p(x_numeric), "--", alpha=0.7, color='red', linewidth=2, label='Trend')
                
                plt.title('Gross Margin Trend Analysis', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Month', fontsize=14)
                plt.ylabel('Gross Margin (%)', fontsize=14)
                plt.legend()
                plt.xticks(rotation=45)
                
                # Add current margin as text
                current_margin = recent_margin['gross_margin_pct'].iloc[-1]
                ax.text(0.02, 0.98, f'Current Margin: {current_margin:.1f}%', 
                       transform=ax.transAxes, fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Insufficient data for margin calculation', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            ax.text(0.5, 0.5, 'Revenue or COGS data missing', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    else:
        ax.text(0.5, 0.5, 'Financial data not available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        plt.title('Gross Margin Analysis', fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
except Exception as e:
    ax.text(0.5, 0.5, f'Chart Error: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    plt.title('Margin Analysis', fontsize=16, fontweight='bold', pad=20)
'''

    # General trend/time series chart
    else:
        return '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    actuals_df = dataframes.get('actuals', pd.DataFrame())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    if not actuals_df.empty:
        data = actuals_df.copy()
        
        if 'month' in data.columns and 'amount' in data.columns:
            data['month'] = pd.to_datetime(data['month'])
            
            # Group by month and sum all amounts
            monthly_totals = data.groupby('month')['amount'].sum().reset_index()
            monthly_totals = monthly_totals.sort_values('month').tail(12)  # Last 12 months
            
            if len(monthly_totals) > 0:
                ax.plot(monthly_totals['month'], monthly_totals['amount'], 
                       marker='o', linewidth=3, markersize=8, color='#1f77b4')
                
                plt.title('Financial Trend Analysis', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Month', fontsize=14)
                plt.ylabel('Amount (USD)', fontsize=14)
                plt.xticks(rotation=45)
                
                # Format y-axis
                max_val = monthly_totals['amount'].max()
                if max_val >= 1000000:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.1f}M'))
                else:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            else:
                ax.text(0.5, 0.5, 'No time series data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        else:
            ax.text(0.5, 0.5, 'Data structure not suitable for charting', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    else:
        ax.text(0.5, 0.5, 'No data available for visualization', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        plt.title('Financial Analysis', fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
except Exception as e:
    ax.text(0.5, 0.5, f'Chart Error: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    plt.title('Financial Chart', fontsize=16, fontweight='bold', pad=20)
'''

# ========== STREAMLIT UI SETUP ========== #
st.set_page_config(page_title="CFO Copilot", layout="wide")
st.title("💼 CFO Copilot")
st.markdown("Upload your financial Excel file and ask questions about revenue, expenses, margins, and cash flow!")

# ========== SESSION STATE INITIALIZATION ========== #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========== FILE UPLOAD (Excel + CSV Support) ========== #
uploaded_file = st.file_uploader(
    "📊 Upload your financial data file", 
    type=["xlsx", "csv"],
    help="Upload Excel file with multiple sheets (actuals, budget, fx, cash) or individual CSV files"
)

if uploaded_file:
    try:
        # ========== PROCESS UPLOADED FILE ========== #
        if uploaded_file.name.endswith('.xlsx'):
            # Process Excel file with multiple sheets
            dataframes, file_info = process_excel_file(uploaded_file)
            st.success("✅ Excel file with multiple sheets loaded successfully!")
            
        else:
            # Process single CSV file (fallback)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            df = pd.read_csv(tmp_path)
            file_name = uploaded_file.name.lower().replace('.csv', '')
            dataframes = {file_name: df}
            file_info = [f"📋 {file_name}: {len(df)} records"]
            st.success("✅ CSV file loaded successfully!")
        
        st.markdown("**Loaded datasets:** " + " | ".join(file_info))
        
        # ========== LLM INITIALIZATION ========== #
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)
        
        # ========== DATA PREPROCESSING FOR LARGE DATASETS ========== #
        def preprocess_large_datasets(dataframes_dict):
            """Optimize datasets for analysis - reduce size while preserving key insights"""
            processed = {}
            for name, df in dataframes_dict.items():
                if len(df) > 10000:  # Very large datasets
                    st.info(f"📊 Large dataset detected for {name.upper()} ({len(df):,} rows). Optimizing for performance...")
                    
                    # For time series data, aggregate by month/quarter for trend analysis
                    if 'month' in df.columns:
                        # Keep recent data (last 24 months) for detailed analysis
                        df['month'] = pd.to_datetime(df['month'])
                        recent_cutoff = df['month'].max() - pd.DateOffset(months=24)
                        recent_data = df[df['month'] >= recent_cutoff]
                        
                        # For older data, aggregate by quarter to reduce size
                        older_data = df[df['month'] < recent_cutoff]
                        if len(older_data) > 0:
                            older_data['quarter'] = older_data['month'].dt.to_period('Q')
                            aggregated_old = older_data.groupby(['quarter', 'entity', 'account_category']).agg({
                                'amount': 'sum' if 'amount' in older_data.columns else 'mean',
                                'cash_usd': 'mean' if 'cash_usd' in older_data.columns else 'sum'
                            }).reset_index()
                            aggregated_old['month'] = aggregated_old['quarter'].dt.start_time
                            aggregated_old = aggregated_old.drop('quarter', axis=1)
                            
                            # Combine recent detailed data with aggregated historical data  
                            processed[name] = pd.concat([aggregated_old, recent_data], ignore_index=True)
                        else:
                            processed[name] = recent_data
                    else:
                        # For non-time series, sample the data
                        processed[name] = df.sample(n=min(5000, len(df)), random_state=42)
                    
                    st.success(f"✅ Optimized {name.upper()}: {len(df):,} → {len(processed[name]):,} rows")
                else:
                    processed[name] = df
            return processed
        
        # Apply preprocessing for large datasets
        if any(len(df) > 10000 for df in dataframes.values()):
            with st.spinner("🔄 Optimizing large datasets for better performance..."):
                dataframes = preprocess_large_datasets(dataframes)
        
        combined_context = ""
        for name, df in dataframes.items():
            combined_context += f"\n{name.upper()} DATA ({len(df)} rows): {', '.join(df.columns)}"

        # ========== CSV ANALYZER FOR LARGE DATASETS ========== #
        def analyze_csv_query(q):
            # For large datasets, provide summary statistics and sampling info
            dataset_summaries = {}
            for name, df in dataframes.items():
                if len(df) > 5000:  # Large dataset
                    # Create summary for large datasets
                    summary = {
                        'total_rows': len(df),
                        'date_range': f"{df['month'].min()} to {df['month'].max()}" if 'month' in df.columns else 'Unknown',
                        'sample_rows': df.head(1000),  # Use sample for analysis
                        'key_columns': list(df.columns),
                        'entities': df['entity'].unique().tolist() if 'entity' in df.columns else [],
                        'account_categories': df['account_category'].unique().tolist() if 'account_category' in df.columns else []
                    }
                    dataset_summaries[name] = summary
                else:
                    # For smaller datasets, use full data
                    dataset_summaries[name] = {'full_data': df, 'total_rows': len(df)}

            # Create agent with optimized data handling
            csv_agent = create_pandas_dataframe_agent(
                llm, 
                # For large datasets, use sampled data; for small datasets, use full data
                [summary.get('sample_rows', summary.get('full_data', df)).head(2000) 
                 for summary, df in zip(dataset_summaries.values(), dataframes.values())],
                verbose=True, 
                handle_parsing_errors=True, 
                allow_dangerous_code=True,
                prefix=f"""
                You are a CFO's financial analysis assistant working with financial data from Excel sheets.
                
                DATASET OVERVIEW:
                {chr(10).join([f"- {name.upper()}: {summary.get('total_rows', 0)} total rows, Date Range: {summary.get('date_range', 'Unknown')}" for name, summary in dataset_summaries.items()])}
                
                IMPORTANT PERFORMANCE NOTES:
                - You are working with sampled data (first 1000-2000 rows per dataset) for performance
                - When providing totals or aggregations, note this is based on sample data
                - For precise financial calculations, focus on specific periods or entities
                - Always mention if analysis is based on sample data for large datasets
                
                Data Structure:
                - ACTUALS: Monthly actual results (month, entity, account_category, amount, currency)
                - BUDGET: Monthly budgeted amounts (month, entity, account_category, amount, currency)  
                - CASH: Monthly cash balances (month, entity, cash_usd)
                - FX: Exchange rates (month, currency, rate_to_usd)
                
                Available Entities: {', '.join(dataset_summaries.get('actuals', {}).get('entities', [])[:5])}
                Available Categories: {', '.join(dataset_summaries.get('actuals', {}).get('account_categories', [])[:10])}
                
                Your role is to provide CFO-level financial insights:
                - Revenue vs Budget variance analysis (filter account_category='Revenue')
                - Gross margin calculation ((Revenue - COGS) / Revenue * 100)
                - Operating expense analysis (filter account_category contains 'Opex')
                - Cash runway and burn rate analysis
                - Period-over-period comparisons
                - Entity performance analysis
                
                For LARGE DATASETS (>5000 rows):
                - Use efficient pandas operations (.loc, .query, .groupby)
                - Sample data when appropriate for trend analysis
                - Focus on specific time periods or entities to reduce data size
                - Always mention data limitations in your response
                
                Analysis Approach:
                - Filter by account_category for specific metrics
                - Group by month for trends, by entity for performance comparison
                - Calculate variances as (Actual - Budget) / Budget * 100
                - Use FX rates for currency conversion when needed
                
                Provide clear, actionable insights suitable for board reporting.
                """
            )
            return csv_agent.run(q)

        # ========== SIMPLIFIED AND RELIABLE FINANCIAL CALCULATOR ========== #
        def efficient_financial_calculator(query):
            """Handle common CFO calculations efficiently for large datasets"""
            try:
                query_lower = query.lower()
                
                # Revenue vs Budget Analysis
                if 'revenue' in query_lower and ('budget' in query_lower or 'vs' in query_lower):
                    actuals = dataframes.get('actuals')
                    budget = dataframes.get('budget') 
                    
                    if actuals is not None and budget is not None:
                        # Convert month to datetime for both dataframes
                        actuals = actuals.copy()
                        budget = budget.copy()
                        actuals['month'] = pd.to_datetime(actuals['month'])
                        budget['month'] = pd.to_datetime(budget['month'])
                        
                        # Extract specific month if mentioned
                        target_month = None
                        if 'june 2025' in query_lower:
                            target_month = pd.to_datetime('2025-06-01')
                        elif 'may 2025' in query_lower:
                            target_month = pd.to_datetime('2025-05-01')
                        elif 'july 2025' in query_lower:
                            target_month = pd.to_datetime('2025-07-01')
                        
                        # Filter for Revenue category
                        actual_revenue = actuals[actuals['account_category'].str.contains('Revenue', case=False, na=False)]
                        budget_revenue = budget[budget['account_category'].str.contains('Revenue', case=False, na=False)]
                        
                        if target_month is not None:
                            # Specific month analysis
                            actual_month = actual_revenue[actual_revenue['month'].dt.to_period('M') == target_month.to_period('M')]
                            budget_month = budget_revenue[budget_revenue['month'].dt.to_period('M') == target_month.to_period('M')]
                            
                            actual_rev = actual_month['amount'].sum()
                            budget_rev = budget_month['amount'].sum()
                            month_name = target_month.strftime('%B %Y')
                        else:
                            # Latest month analysis
                            latest_month = actual_revenue['month'].max()
                            actual_month = actual_revenue[actual_revenue['month'] == latest_month]
                            budget_month = budget_revenue[budget_revenue['month'] == latest_month]
                            
                            actual_rev = actual_month['amount'].sum()
                            budget_rev = budget_month['amount'].sum()
                            month_name = latest_month.strftime('%B %Y')
                        
                        if budget_rev > 0:
                            variance_abs = actual_rev - budget_rev
                            variance_pct = (variance_abs / budget_rev) * 100
                            
                            return f"""
**Revenue vs Budget Analysis - {month_name}:**

• **Actual Revenue**: ${actual_rev:,.0f}
• **Budget Revenue**: ${budget_rev:,.0f} 
• **Variance**: ${variance_abs:,.0f} ({variance_pct:+.1f}%)

**Business Context**: {'✅ Favorable - Exceeded budget' if variance_pct > 0 else '⚠️ Unfavorable - Below budget'} by {abs(variance_pct):.1f}%

**Key Insights**:
- {"Strong performance above target" if variance_pct > 2 else "Close to budget target" if abs(variance_pct) <= 2 else "Significant variance requiring attention"}
- Recommend {"celebrating success and analyzing drivers" if variance_pct > 0 else "investigating shortfall causes"}
                            """
                        else:
                            return f"No budget data found for revenue analysis in {month_name}"
                    else:
                        return "Revenue vs Budget analysis requires both actuals and budget datasets"
                
                # Cash Runway Analysis
                elif 'cash runway' in query_lower or ('cash' in query_lower and 'runway' in query_lower):
                    cash_data = dataframes.get('cash')
                    if cash_data is not None:
                        cash_df = cash_data.copy()
                        cash_df['month'] = pd.to_datetime(cash_df['month'])
                        cash_df = cash_df.sort_values('month')
                        
                        if len(cash_df) >= 3:
                            # Get last 3 months for burn calculation
                            recent_cash = cash_df.tail(3)
                            monthly_changes = recent_cash['cash_usd'].diff().dropna()
                            avg_monthly_burn = -monthly_changes.mean() if monthly_changes.mean() < 0 else 0
                            
                            current_cash = cash_df['cash_usd'].iloc[-1]
                            runway_months = current_cash / avg_monthly_burn if avg_monthly_burn > 0 else float('inf')
                            
                            return f"""
**Cash Runway Analysis:**

• **Current Cash Balance**: ${current_cash:,.0f}
• **Average Monthly Burn**: ${avg_monthly_burn:,.0f}
• **Estimated Runway**: {runway_months:.1f} months

**Risk Assessment**: 
{
'🔴 CRITICAL - Less than 6 months runway' if runway_months < 6 else
'🟡 CAUTION - Monitor closely' if runway_months < 12 else
'🟢 HEALTHY - Sufficient runway' if runway_months < 24 else
'🟢 EXCELLENT - Strong cash position'
}

**Actionable Insights**:
- {"Immediate action required to reduce burn or raise capital" if runway_months < 6 else "Plan for fundraising or cost optimization" if runway_months < 12 else "Maintain current cash management practices"}
                            """
                        else:
                            return "Insufficient cash data (need at least 3 months) for runway calculation"
                    else:
                        return "Cash runway analysis requires cash balance data"
                
                # If not a common query, return None to let other tools handle it
                return None
                
            except Exception as e:
                return f"Error in financial calculation: {str(e)}"

        # ========== SIMPLIFIED TOOLS WITH CLEAR DESCRIPTIONS ========== #
        tools = [
            Tool.from_function(
                efficient_financial_calculator,
                name="CFO_QuickAnalysis", 
                description="Use this for common CFO questions: 'revenue vs budget', 'cash runway', 'June 2025 revenue'. Returns formatted financial analysis.",
            ),
            Tool.from_function(
                PythonREPLTool().run, 
                name="CFO_Calculator", 
                description="Use for complex calculations, custom metrics, or when CFO_QuickAnalysis returns None. Can do math, data manipulation.",
                allow_dangerous_code=True
            ),
            Tool.from_function(
                analyze_csv_query, 
                name="CFO_DataAnalyzer", 
                description="Use for detailed data exploration, filtering, grouping, or when other tools fail. Handles complex financial dataset queries."
            )
        ]

        # ========== AGENT INITIALIZATION WITH BETTER ERROR HANDLING ========== #
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,  # This is crucial for handling LLM output parsing errors
            max_iterations=3,  # Limit iterations to avoid infinite loops
            early_stopping_method="generate",  # Stop early if agent gets confused
            return_intermediate_steps=True  # Help with debugging
        )

        # ========== DISPLAY CHAT HISTORY ========== #
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ========== CHAT INPUT ========== #
        user_input = st.chat_input("Ask about revenue vs budget, margins, opex breakdown, cash runway, entity performance...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing financial data across all datasets..."):
                    trace_output = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(trace_output):
                            # CFO-focused query construction
                            dataset_summary = " | ".join([f"{k}: {len(v)} rows" for k, v in dataframes.items()])
                            
                            # Smart query routing for large datasets
                            total_rows = sum(len(df) for df in dataframes.values())
                            is_large_dataset = total_rows > 20000
                            
                            if is_large_dataset:
                                performance_note = f"📊 Working with large dataset ({total_rows:,} total rows). Using optimized analysis methods."
                                st.info(performance_note)
                            
                            query = f"""
                            You are analyzing financial data for a CFO using {'optimized' if is_large_dataset else 'complete'} Excel datasets: {dataset_summary}
                            
                            {'PERFORMANCE NOTE: Large dataset detected. Use EfficientFinancialCalculator for common metrics like revenue vs budget and cash runway.' if is_large_dataset else ''}
                            
                            The data includes:
                            - ACTUALS vs BUDGET comparison (month, entity, account_category, amount, currency)
                            - CASH flow analysis (monthly cash balances by entity)  
                            - FX RATES for currency conversion
                            {'- Data has been optimized: recent 24 months detailed, older data aggregated by quarter' if is_large_dataset else ''}
                            
                            CFO Question: {user_input}
                            
                            TOOL SELECTION GUIDANCE:
                            - Use EfficientFinancialCalculator for: revenue vs budget, cash runway, margin analysis
                            - Use FinancialDataAnalyzer for: complex filtering, entity comparisons, trend analysis
                            - Use AdvancedFinancialCalculator for: custom calculations, specialized metrics
                            
                            Please provide:
                            1. Direct numerical answer with key financial metrics
                            2. Business context on what this means for financial performance
                            3. Notable trends, variances, or risks to highlight  
                            4. Actionable insights for CFO decision-making
                            {'5. Note any data limitations due to large dataset optimization' if is_large_dataset else ''}
                            
                            Focus on:
                            - Actual vs Budget variances (calculate percentage differences)
                            - Revenue trends and margin analysis
                            - Cash flow and runway implications
                            - Entity-level performance differences
                            - Month-over-month trends
                            """
                            response = agent.invoke(query)
                        reply = response["output"]
                    except Exception as e:
                        reply = f"⚠️ Error analyzing financial data: {str(e)}"

                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

                    with st.expander("🧠 Financial Analysis Trace", expanded=False):
                        st.code(trace_output.getvalue())

                # ========== CHART OUTPUT ========== #
                if is_graph_query(user_input):
                    try:
                        st.markdown("---")
                        st.markdown("### 📊 Executive Chart")
                        chart_code = generate_chart_code(llm, dataframes, user_input)
                        
                        with st.expander("📝 Generated Chart Code", expanded=False):
                            st.code(chart_code, language="python")
                        
                        fig = safe_exec_chart_code(chart_code, dataframes)
                        if fig:
                            st.pyplot(fig)
                            st.caption("Executive-ready chart generated for CFO presentation.")
                        else:
                            st.warning("Chart generated but could not be displayed. Check the code above.")
                            
                    except Exception as e:
                        st.error(f"⚠️ Could not generate financial chart: {e}")
                        st.markdown("💡 Try rephrasing your request or check if the required data columns exist.")

        # ========== SIDEBAR WITH DATA PREVIEW AND PERFORMANCE INFO ========== #
        with st.sidebar:
            st.header("📋 Data Overview")
            
            # Dataset size summary
            total_rows = sum(len(df) for df in dataframes.values())
            if total_rows > 50000:
                st.error(f"⚠️ Very Large Dataset: {total_rows:,} total rows")
                st.markdown("*Data has been optimized for performance*")
            elif total_rows > 20000:
                st.warning(f"📊 Large Dataset: {total_rows:,} total rows")
                st.markdown("*Using optimized analysis methods*")
            else:
                st.success(f"✅ Dataset Size: {total_rows:,} total rows")
            
            for name, df in dataframes.items():
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
                    # Store the selected question to be processed
                    st.session_state.selected_question = question
                    st.rerun()
        
        # ========== HANDLE SELECTED SAMPLE QUESTION ========== #
        if hasattr(st.session_state, 'selected_question'):
            # Process the selected question as if it was typed in chat
            question = st.session_state.selected_question
            del st.session_state.selected_question  # Remove it so it doesn't repeat
            
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing financial data across all datasets..."):
                    trace_output = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(trace_output):
                            dataset_summary = " | ".join([f"{k}: {len(v)} rows" for k, v in dataframes.items()])
                            
                            query = f"""
                            You are analyzing financial data for a CFO using an Excel file with multiple sheets: {dataset_summary}
                            
                            The data includes:
                            - ACTUALS vs BUDGET comparison capability (same structure: month, entity, account_category, amount, currency)
                            - CASH flow analysis (monthly cash balances by entity)
                            - FX RATES for currency conversion
                            
                            CFO Question: {question}
                            
                            Please provide:
                            1. Direct numerical answer with key financial metrics
                            2. Business context on what this means for financial performance
                            3. Notable trends, variances, or risks to highlight  
                            4. Actionable insights for CFO decision-making
                            
                            Focus on:
                            - Actual vs Budget variances (calculate percentage differences)
                            - Revenue trends and margin analysis
                            - Cash flow and runway implications
                            - Entity-level performance differences
                            - Month-over-month trends
                            """
                            response = agent.invoke(query)
                        reply = response["output"]
                    except Exception as e:
                        reply = f"⚠️ Error analyzing financial data: {str(e)}"

                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})

                    with st.expander("🧠 Financial Analysis Trace", expanded=False):
                        st.code(trace_output.getvalue())

                # Generate chart if it's a visualization question
                if is_graph_query(question):
                    try:
                        st.markdown("---")
                        st.markdown("### 📊 Executive Chart")
                        chart_code = generate_chart_code(llm, dataframes, question)
                        
                        with st.expander("📝 Generated Chart Code", expanded=False):
                            st.code(chart_code, language="python")
                        
                        fig = safe_exec_chart_code(chart_code, dataframes)
                        if fig:
                            st.pyplot(fig)
                            st.caption("Executive-ready chart generated for CFO presentation.")
                        else:
                            st.warning("Chart generated but could not be displayed. Check the code above.")
                            
                    except Exception as e:
                        st.error(f"⚠️ Could not generate financial chart: {e}")
                        st.markdown("💡 Try rephrasing your request or check if the required data columns exist.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.markdown("Please ensure your file is a valid Excel (.xlsx) or CSV file.")

# ========== WHEN NO FILE IS UPLOADED ========== #
else:
    st.info("Please upload your financial Excel file or CSV to begin CFO analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📁 **Expected File Structure:**
        
        **Excel File (.xlsx) with sheets:**
        - `actuals` - Monthly actual results
        - `budget` - Monthly budget data  
        - `cash` - Cash balances
        - `fx` - Exchange rates
        
        **Or individual CSV files**
        """)
        
        st.markdown("""
        ### 📊 **Data Columns Expected:**
        - **Actuals/Budget**: month, entity, account_category, amount, currency
        - **Cash**: month, entity, cash_usd
        - **FX**: month, currency, rate_to_usd
        """)
    
    with col2:
        st.markdown("""
        ### 💼 **Sample CFO Questions:**
        - "What was June 2025 revenue vs budget?"
        - "Show revenue trends by month for 2024"
        - "Break down operating expenses by category"  
        - "What is our cash runway based on burn rate?"
        - "Compare entity performance across divisions"
        - "Calculate gross margin trends over time"
        - "Show budget variances by account category"
        """)