import pandas as pd
import streamlit as st
import contextlib
import io
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonREPLTool

class FinancialAnalyzer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)
        self.tools = None
        self.agent = None
    
    def analyze_query(self, query, dataframes_dict):
        """Main analysis method using agent with tools"""
        try:
            # Initialize tools and agent
            self._setup_tools(dataframes_dict)
            
            # CFO-focused query construction
            dataset_summary = " | ".join([f"{k}: {len(v)} rows" for k, v in dataframes_dict.items()])
            total_rows = sum(len(df) for df in dataframes_dict.values())
            is_large_dataset = total_rows > 20000
            
            if is_large_dataset:
                performance_note = f"📊 Working with large dataset ({total_rows:,} total rows). Using optimized analysis methods."
                st.info(performance_note)
            
            cfo_query = f"""
            You are analyzing financial data for a CFO using {'optimized' if is_large_dataset else 'complete'} Excel datasets: {dataset_summary}
            
            {'PERFORMANCE NOTE: Large dataset detected. Use CFO_QuickAnalysis for common metrics like revenue vs budget and cash runway.' if is_large_dataset else ''}
            
            The data includes:
            - ACTUALS vs BUDGET comparison (month, entity, account_category, amount, currency)
            - CASH flow analysis (monthly cash balances by entity)  
            - FX RATES for currency conversion
            {'- Data has been optimized: recent 24 months detailed, older data aggregated by quarter' if is_large_dataset else ''}
            
            CFO Question: {query}
            
            TOOL SELECTION GUIDANCE:
            - Use CFO_QuickAnalysis for: revenue vs budget, cash runway, margin analysis
            - Use CFO_DataAnalyzer for: complex filtering, entity comparisons, trend analysis
            - Use CFO_Calculator for: custom calculations, specialized metrics
            
            Please provide:
            1. Direct numerical answer with key financial metrics
            2. Business context on what this means for financial performance
            3. Notable trends, variances, or risks to highlight  
            4. Actionable insights for CFO decision-making
            {'5. Note any data limitations due to large dataset optimization' if is_large_dataset else ''}
            """
            
            # Execute with agent
            trace_output = io.StringIO()
            with contextlib.redirect_stdout(trace_output):
                response = self.agent.invoke(cfo_query)
            
            return response["output"]
            
        except Exception as e:
            return f"⚠️ Error analyzing financial data: {str(e)}"
    
    def _setup_tools(self, dataframes_dict):
        """Setup financial analysis tools"""
        self.tools = [
            Tool.from_function(
                lambda q: self._quick_financial_analysis(q, dataframes_dict),
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
                lambda q: self._analyze_csv_query(q, dataframes_dict), 
                name="CFO_DataAnalyzer", 
                description="Use for detailed data exploration, filtering, grouping, or when other tools fail. Handles complex financial dataset queries."
            )
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate",
            return_intermediate_steps=True
        )
    
    def _quick_financial_analysis(self, query, dataframes_dict):
        """Handle common CFO calculations efficiently"""
        try:
            query_lower = query.lower()
            
            # Revenue vs Budget Analysis
            if 'revenue' in query_lower and ('budget' in query_lower or 'vs' in query_lower):
                return self._analyze_revenue_vs_budget(query_lower, dataframes_dict)
            
            # Cash Runway Analysis
            elif 'cash runway' in query_lower or ('cash' in query_lower and 'runway' in query_lower):
                return self._analyze_cash_runway(dataframes_dict)
            
            # Return None to let other tools handle it
            return None
            
        except Exception as e:
            return f"Error in financial calculation: {str(e)}"
    
    def _analyze_revenue_vs_budget(self, query, dataframes_dict):
        """Analyze revenue vs budget with real data"""
        actuals = dataframes_dict.get('actuals')
        budget = dataframes_dict.get('budget')
        
        if actuals is None or budget is None:
            return "Revenue vs Budget analysis requires both actuals and budget datasets"
        
        try:
            # Convert month to datetime for both dataframes
            actuals = actuals.copy()
            budget = budget.copy()
            actuals['month'] = pd.to_datetime(actuals['month'])
            budget['month'] = pd.to_datetime(budget['month'])
            
            # Extract specific month if mentioned
            target_month = None
            if 'june 2025' in query:
                target_month = pd.to_datetime('2025-06-01')
            elif 'may 2025' in query:
                target_month = pd.to_datetime('2025-05-01')
            elif 'july 2025' in query:
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
                
        except Exception as e:
            return f"Error in revenue analysis: {str(e)}"
    
    def _analyze_cash_runway(self, dataframes_dict):
        """Analyze cash runway with real data"""
        cash_data = dataframes_dict.get('cash')
        if cash_data is None:
            return "Cash runway analysis requires cash balance data"
        
        try:
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
                
        except Exception as e:
            return f"Error in cash analysis: {str(e)}"
    
    def _analyze_csv_query(self, query, dataframes_dict):
        """Detailed data analysis using pandas agent"""
        # For large datasets, provide summary statistics and sampling info
        dataset_summaries = {}
        for name, df in dataframes_dict.items():
            if len(df) > 5000:  # Large dataset
                summary = {
                    'total_rows': len(df),
                    'date_range': f"{df['month'].min()} to {df['month'].max()}" if 'month' in df.columns else 'Unknown',
                    'sample_rows': df.head(1000),
                    'key_columns': list(df.columns),
                    'entities': df['entity'].unique().tolist() if 'entity' in df.columns else [],
                    'account_categories': df['account_category'].unique().tolist() if 'account_category' in df.columns else []
                }
                dataset_summaries[name] = summary
            else:
                dataset_summaries[name] = {'full_data': df, 'total_rows': len(df)}

        # Create agent with optimized data handling
        csv_agent = create_pandas_dataframe_agent(
            self.llm, 
            [summary.get('sample_rows', summary.get('full_data', df)).head(2000) 
             for summary, df in zip(dataset_summaries.values(), dataframes_dict.values())],
            verbose=True, 
            handle_parsing_errors=True, 
            allow_dangerous_code=True,
            prefix=f"""
            You are a CFO's financial analysis assistant working with financial data from Excel sheets.
            
            DATASET OVERVIEW:
            {chr(10).join([f"- {name.upper()}: {summary.get('total_rows', 0)} total rows, Date Range: {summary.get('date_range', 'Unknown')}" for name, summary in dataset_summaries.items()])}
            
            Your role is to provide CFO-level financial insights:
            - Revenue vs Budget variance analysis (filter account_category='Revenue')
            - Gross margin calculation ((Revenue - COGS) / Revenue * 100)
            - Operating expense analysis (filter account_category contains 'Opex')
            - Cash runway and burn rate analysis
            - Period-over-period comparisons
            - Entity performance analysis
            
            Analysis Approach:
            - Filter by account_category for specific metrics
            - Group by month for trends, by entity for performance comparison
            - Calculate variances as (Actual - Budget) / Budget * 100
            - Use FX rates for currency conversion when needed
            
            Provide clear, actionable insights suitable for board reporting.
            """
        )
        return csv_agent.run(query)