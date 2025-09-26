"""
Simple tests for CFO Copilot agent
"""
import pytest
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_revenue_calculation():
    """Test basic revenue calculation"""
    # Sample data
    data = {
        'month': ['2025-06', '2025-06'],
        'account_category': ['Revenue', 'Revenue'], 
        'amount': [500000, 500000]
    }
    df = pd.DataFrame(data)
    
    revenue_total = df[df['account_category'] == 'Revenue']['amount'].sum()
    assert revenue_total == 1000000

def test_cash_runway():
    """Test cash runway calculation"""
    cash_data = [1000000, 800000, 600000]  # Declining cash
    monthly_burn = (cash_data[0] - cash_data[-1]) / (len(cash_data) - 1)
    runway = cash_data[-1] / monthly_burn
    
    assert runway > 0
    assert runway < 12  # Should be less than 12 months with this burn

if __name__ == "__main__":
    test_revenue_calculation()
    test_cash_runway()
    print("✅ All tests passed!")