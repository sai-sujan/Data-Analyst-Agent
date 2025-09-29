import tempfile
import pandas as pd
import streamlit as st

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

def get_data_summary(dataframes_dict):
    """Generate a summary of loaded datasets"""
    summary = {}
    for name, df in dataframes_dict.items():
        info = {
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': None,
            'entities': [],
            'categories': []
        }
        
        # Date range
        if 'month' in df.columns:
            try:
                dates = pd.to_datetime(df['month'], errors='coerce').dropna()
                if not dates.empty:
                    info['date_range'] = f"{dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}"
            except:
                pass
        
        # Unique entities
        if 'entity' in df.columns:
            info['entities'] = df['entity'].unique().tolist()[:10]
        
        # Account categories
        if 'account_category' in df.columns:
            info['categories'] = df['account_category'].unique().tolist()[:10]
        
        summary[name] = info
    
    return summary