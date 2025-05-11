from src.rag_db import get_vector_store
import json
from datetime import datetime, date
import pandas as pd

def format_record(record):
    """Format a record for pretty printing."""
    # Create a safe version of the record that won't raise KeyError
    formatted = {
        'ticker': record.get('ticker', 'N/A'),
        'week_start': record.get('week_start', 'N/A').strftime('%Y-%m-%d') if isinstance(record.get('week_start'), date) else 'N/A',
        'features': record.get('features', {}),
        'texts': record.get('texts', []),
        'embeddings': f"Vector of length {len(record.get('embeddings', []))}" if 'embeddings' in record else 'N/A',
        'outcome': record.get('outcome', 'N/A')
    }
    
    return formatted

def main():
    # Initialize vector store
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    
    # Get all records
    records = vector_store.get_all()
    
    print(f"\nFound {len(records)} total records in the database")
    
    # Filter for 2011 records
    records_2011 = [r for r in records if isinstance(r.get('week_start'), date) and r.get('week_start').year == 2011]
    print(f"\nFound {len(records_2011)} records from 2011")
    
    # Sort by date
    records_2011.sort(key=lambda x: x.get('week_start'))
    
    # Display first 5 records from 2011
    print("\nSample Records from 2011:")
    for i, record in enumerate(records_2011[:5]):
        print(f"\nRecord {i+1}:")
        formatted_record = format_record(record)
        print(json.dumps(formatted_record, indent=2))
        
        # Print available features
        if 'features' in record:
            print("\nAvailable features:")
            for key, value in record['features'].items():
                if not pd.isna(value):  # Only show non-NaN values
                    print(f"{key}: {value}")
    
    # Display summary statistics
    print("\nDatabase Summary:")
    tickers = set(record.get('ticker', 'UNKNOWN') for record in records)
    print(f"Unique tickers: {tickers}")
    
    # Get date range
    dates = [record.get('week_start') for record in records if isinstance(record.get('week_start'), date)]
    if dates:
        print(f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
    
    # Count records per ticker
    print("\nRecords per ticker:")
    for ticker in tickers:
        count = sum(1 for record in records if record.get('ticker') == ticker)
        print(f"{ticker}: {count} records")

if __name__ == "__main__":
    main() 