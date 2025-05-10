from src.ingestion.data_fetcher import DataFetcher

fetcher = DataFetcher()
try:
    df = fetcher.fetch_stock_data("AAPL", "2022-06-01", "2022-06-30")
    print("Fetched DataFrame shape:", df.shape)
    print(df.tail(5))  # Show the last 5 rows

    if df.empty:
        print("DataFrame is empty!")
    else:
        print("Last row as dict:", df.iloc[-1].to_dict())
except Exception as e:
    print(f"Exception occurred: {e}") 