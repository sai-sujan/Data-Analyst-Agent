import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def is_graph_query(query: str) -> bool:
    """Detect if CFO query requests visualization"""
    keywords = [
        "plot", "graph", "chart", "show", "visualize", "trend", 
        "breakdown", "comparison", "over time", "by month", "dashboard"
    ]
    return any(k in query.lower() for k in keywords)

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

    # General trend/time series chart
    else:
        return '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    actuals_df = dataframes.get('actuals', pd.DataFrame())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
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