from datetime import date
from src.prompt_builder import build_prediction_prompt

if __name__ == '__main__':
    # Real data for AAPL (example)
    current_features = {
        'rsi': 45,
        'pe_ratio': 25,
        'open_price': 150,
        'close_price': 155,
        'high_price': 160,
        'low_price': 145
    }
    current_texts = [
        "Apple announces new product.",
        "Market sentiment is bullish."
    ]
    prompt = build_prediction_prompt('AAPL', date(2023, 1, 1), current_features, current_texts)
    print(prompt) 