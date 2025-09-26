"""
Simple CFO analysis tools
"""
import pandas as pd

def calculate_variance(actual, budget):
    """Calculate variance between actual and budget"""
    if budget == 0:
        return 0
    return ((actual - budget) / budget) * 100

def calculate_gross_margin(revenue, cogs):
    """Calculate gross margin percentage"""
    if revenue == 0:
        return 0
    return ((revenue - cogs) / revenue) * 100

def calculate_cash_runway(cash_balances):
    """Calculate cash runway in months"""
    if len(cash_balances) < 2:
        return 0
    
    monthly_burn = abs(cash_balances[-1] - cash_balances[-2])
    if monthly_burn == 0:
        return float('inf')
    
    return cash_balances[-1] / monthly_burn

def filter_by_category(df, category):
    """Filter dataframe by account category"""
    return df[df['account_category'].str.contains(category, case=False, na=False)]

def get_latest_month_data(df):
    """Get data for the latest month"""
    latest_month = df['month'].max()
    return df[df['month'] == latest_month]

def format_currency(amount):
    """Format amount as currency"""
    if amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"${amount/1000:.0f}K"
    else:
        return f"${amount:,.0f}"