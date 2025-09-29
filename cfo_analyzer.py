import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

class CFOAnalyzer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    
    def analyze_query(self, query, dataframes_dict):
        """Main analysis method that uses only real data with Gemini API"""
        try:
            # Always use pandas agent with Gemini for dynamic analysis
            return self._analyze_with_pandas_agent(query, dataframes_dict)
                
        except Exception as e:
            return f"Analysis error: {str(e)}\n\nPlease check your data format and try again."
    
    def _analyze_with_pandas_agent(self, query, dataframes_dict):
        """Use pandas agent with Gemini for all queries - no hardcoding"""
        if not dataframes_dict:
            return "No data available for analysis. Please upload your financial data files."
        
        # Select the most relevant dataset for the query
        main_df = self._select_relevant_dataframe(dataframes_dict, query)
        
        if main_df is None or main_df.empty:
            return "No relevant data found for your query. Please check your data files."
        
        # Limit size for performance but use real data
        original_size = len(main_df)
        if len(main_df) > 1500:
            main_df = main_df.head(1500)
            st.info(f"Analyzing first 1,500 rows from {original_size:,} total rows for performance")
        
        try:
            agent = create_pandas_dataframe_agent(
                self.llm,
                main_df,
                verbose=False,
                handle_parsing_errors=True,
                allow_dangerous_code=True,
                prefix=f"""
You are a CFO financial analyst. You must analyze the ACTUAL DATA in the provided dataframe.

DATASET INFO:
- Rows: {len(main_df)}
- Columns: {list(main_df.columns)}
- Sample data preview: {main_df.head(2).to_string()}

CRITICAL INSTRUCTIONS:
1. Use ONLY the real data from this dataframe - NO sample/dummy numbers
2. For revenue vs budget queries: Filter by account_category containing 'Revenue' 
3. For cash queries: Look for cash_usd or similar cash columns
4. Calculate actual financial metrics from the data
5. Always show specific dollar amounts and percentages from real data
6. If data is missing, clearly state what's missing

QUERY: {query}

Provide a CFO-level analysis using the actual data in the dataframe.
                """
            )
            
            result = agent.run(query)
            return result + "\n\n🤖 *Analysis based on your actual uploaded data*"
            
        except Exception as e:
            # If pandas agent fails, try direct data inspection
            return self._direct_data_analysis(query, main_df, dataframes_dict)
    
    def _select_relevant_dataframe(self, dataframes_dict, query):
        """Select the most relevant dataframe based on query content"""
        query_lower = query.lower()
        
        # For revenue/budget queries, prefer actuals, then budget
        if 'revenue' in query_lower or 'budget' in query_lower:
            if 'actuals' in dataframes_dict:
                return dataframes_dict['actuals']
            elif 'budget' in dataframes_dict:
                return dataframes_dict['budget']
        
        # For cash queries, prefer cash data
        elif 'cash' in query_lower:
            if 'cash' in dataframes_dict:
                return dataframes_dict['cash']
        
        # For expense queries, prefer actuals
        elif any(word in query_lower for word in ['expense', 'opex', 'cost']):
            if 'actuals' in dataframes_dict:
                return dataframes_dict['actuals']
        
        # Default: return the largest dataset
        if dataframes_dict:
            return max(dataframes_dict.values(), key=len)
        
        return None
    
    def _direct_data_analysis(self, query, main_df, dataframes_dict):
        """Direct analysis when pandas agent fails"""
        try:
            query_lower = query.lower()
            
            # Revenue vs Budget analysis
            if 'revenue' in query_lower and 'budget' in query_lower:
                return self._analyze_revenue_direct(query, dataframes_dict)
            
            # Cash runway analysis
            elif 'cash' in query_lower and 'runway' in query_lower:
                return self._analyze_cash_direct(dataframes_dict)
            
            # General data summary
            else:
                return self._provide_data_summary(main_df, query)
                
        except Exception as e:
            return f"Direct analysis failed: {str(e)}"
    
    def _analyze_revenue_direct(self, query, dataframes_dict):
        """Direct revenue analysis using real data"""
        actuals = dataframes_dict.get('actuals')
        budget = dataframes_dict.get('budget')
        
        if actuals is None:
            return "Revenue analysis requires 'actuals' data. Please upload a file with an 'actuals' sheet."
        
        try:
            # Check if revenue data exists
            revenue_data = actuals[actuals['account_category'].str.contains('Revenue', case=False, na=False)]
            
            if revenue_data.empty:
                available_categories = actuals['account_category'].unique() if 'account_category' in actuals.columns else []
                return f"No revenue data found. Available categories: {list(available_categories)[:10]}"
            
            # Extract target month if specified
            target_month = self._extract_target_month(query)
            
            if target_month:
                # Filter for specific month
                month_data = revenue_data[revenue_data['month'].astype(str).str.contains(target_month, na=False)]
                actual_revenue = month_data['amount'].sum() if not month_data.empty else 0
                
                # Get budget data if available
                budget_revenue = 0
                if budget is not None:
                    budget_rev_data = budget[
                        (budget['account_category'].str.contains('Revenue', case=False, na=False)) &
                        (budget['month'].astype(str).str.contains(target_month, na=False))
                    ]
                    budget_revenue = budget_rev_data['amount'].sum() if not budget_rev_data.empty else 0
                
                if budget_revenue > 0:
                    variance = ((actual_revenue - budget_revenue) / budget_revenue) * 100
                    performance = "✅ Above budget" if variance > 0 else "⚠️ Below budget"
                    
                    return f"""
**{target_month.title()} Revenue vs Budget Analysis:**

• **Actual Revenue**: ${actual_revenue:,.0f}
• **Budget Revenue**: ${budget_revenue:,.0f}
• **Variance**: ${actual_revenue - budget_revenue:,.0f} ({variance:+.1f}%)

**Performance**: {performance} by {abs(variance):.1f}%

**Data Source**: Your uploaded financial data
                    """
                else:
                    return f"Found actual revenue of ${actual_revenue:,.0f} for {target_month}, but no budget data available for comparison."
            
            else:
                # Latest month analysis
                latest_month = revenue_data['month'].max()
                latest_revenue = revenue_data[revenue_data['month'] == latest_month]['amount'].sum()
                
                return f"""
**Revenue Analysis - {latest_month}:**

• **Latest Month Revenue**: ${latest_revenue:,.0f}
• **Data Points**: {len(revenue_data)} revenue entries found
• **Date Range**: {revenue_data['month'].min()} to {revenue_data['month'].max()}

**Note**: Budget comparison requires both actuals and budget datasets.
                """
                
        except Exception as e:
            return f"Error analyzing revenue data: {str(e)}"
    
    def _analyze_cash_direct(self, dataframes_dict):
        """Direct cash analysis using real data"""
        cash_data = dataframes_dict.get('cash')
        
        if cash_data is None:
            return "Cash analysis requires 'cash' data. Please upload a file with a 'cash' sheet."
        
        try:
            # Look for cash columns
            cash_columns = [col for col in cash_data.columns if 'cash' in col.lower()]
            
            if not cash_columns:
                return f"No cash columns found. Available columns: {list(cash_data.columns)}"
            
            cash_col = cash_columns[0]  # Use first cash column found
            
            # Sort by month and calculate runway
            cash_df = cash_data.copy()
            cash_df['month'] = pd.to_datetime(cash_df['month'], errors='coerce')
            cash_df = cash_df.dropna(subset=['month']).sort_values('month')
            
            if len(cash_df) < 2:
                return "Need at least 2 months of cash data for runway analysis."
            
            current_cash = cash_df[cash_col].iloc[-1]
            
            # Calculate burn rate
            if len(cash_df) >= 3:
                recent_months = cash_df.tail(3)
                monthly_changes = recent_months[cash_col].diff().dropna()
                avg_change = monthly_changes.mean()
                burn_rate = -avg_change if avg_change < 0 else 0
                
                runway_months = current_cash / burn_rate if burn_rate > 0 else float('inf')
                
                risk_level = ("🔴 CRITICAL" if runway_months < 6 else
                             "🟡 CAUTION" if runway_months < 12 else
                             "🟢 HEALTHY")
                
                return f"""
**Cash Runway Analysis:**

• **Current Cash**: ${current_cash:,.0f}
• **Average Monthly Burn**: ${burn_rate:,.0f}
• **Estimated Runway**: {runway_months:.1f} months

**Risk Assessment**: {risk_level}

**Data Period**: {cash_df['month'].min().strftime('%Y-%m')} to {cash_df['month'].max().strftime('%Y-%m')}
                """
            else:
                return f"Current cash position: ${current_cash:,.0f}. Need more historical data for burn rate calculation."
                
        except Exception as e:
            return f"Error analyzing cash data: {str(e)}"
    
    def _provide_data_summary(self, main_df, query):
        """Provide summary of available data"""
        summary_parts = []
        summary_parts.append(f"**Dataset Overview:**")
        summary_parts.append(f"• Rows: {len(main_df):,}")
        summary_parts.append(f"• Columns: {len(main_df.columns)}")
        
        if 'month' in main_df.columns:
            date_range = f"{main_df['month'].min()} to {main_df['month'].max()}"
            summary_parts.append(f"• Date Range: {date_range}")
        
        if 'account_category' in main_df.columns:
            categories = main_df['account_category'].unique()[:5]
            summary_parts.append(f"• Categories: {', '.join(categories)}")
        
        summary_parts.append(f"\n**Query**: {query}")
        summary_parts.append("\n**Available Analysis**: Try specific questions about revenue, expenses, or cash flow.")
        
        return "\n".join(summary_parts)
    
    def _extract_target_month(self, query):
        """Extract target month from query"""
        month_mapping = {
            'june 2025': '2025-06',
            'may 2025': '2025-05', 
            'july 2025': '2025-07',
            'april 2025': '2025-04',
            'august 2025': '2025-08',
            'september 2025': '2025-09'
        }
        
        query_lower = query.lower()
        for month_text, month_code in month_mapping.items():
            if month_text in query_lower:
                return month_code
        
        return None