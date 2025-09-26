"""
Simple tests for chart generation
"""
import pytest
import pandas as pd
import matplotlib.pyplot as plt

def test_chart_keywords():
    """Test chart keyword detection"""
    chart_keywords = ['show', 'plot', 'chart', 'graph', 'visualize', 'trend']
    
    test_queries = [
        "show me revenue trends",
        "plot cash runway", 
        "chart expenses by category"
    ]
    
    for query in test_queries:
        has_chart_keyword = any(keyword in query.lower() for keyword in chart_keywords)
        assert has_chart_keyword

def test_chart_creation():
    """Test basic chart creation"""
    # Sample data
    data = pd.DataFrame({
        'month': ['2025-01', '2025-02', '2025-03'],
        'revenue': [100000, 120000, 110000]
    })
    
    fig, ax = plt.subplots()
    ax.plot(data['month'], data['revenue'])
    ax.set_title('Revenue Trend')
    
    assert fig is not None
    assert ax.get_title() == 'Revenue Trend'
    
    plt.close(fig)

if __name__ == "__main__":
    test_chart_keywords() 
    test_chart_creation()
    print("✅ Chart tests passed!")