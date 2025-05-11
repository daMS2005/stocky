import os
from datetime import datetime, timedelta
import openai
from src.prompt_builder import build_prediction_prompt
from src.llm_inference import get_llm_prediction, format_prediction_output

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    # Test parameters
    ticker = "AAPL"
    current_date = datetime.now()
    current_week_start = current_date - timedelta(days=current_date.weekday())
    
    # Get the prompt
    prompt = build_prediction_prompt(
        ticker=ticker,
        current_week_start=current_week_start,
        current_features={
            "technical": {
                "rsi": 65.5,
                "macd": 2.3,
                "bollinger_upper": 180.5,
                "bollinger_lower": 170.2,
                "volume": 50000000
            },
            "fundamental": {
                "pe_ratio": 28.5,
                "dividend_yield": 0.6,
                "market_cap": 2800000000000
            },
            "macro": {
                "inflation_rate": 3.2,
                "interest_rate": 5.25,
                "gdp_growth": 2.1
            }
        },
        current_texts=[
            "Apple announces new AI features for iPhone",
            "Strong Q4 earnings beat expectations",
            "New product line expansion planned for 2024"
        ]
    )
    
    print("\nGenerated Prompt:")
    print(prompt)
    
    # Get LLM prediction
    action, reasoning = get_llm_prediction(prompt)
    
    # Format and print the output
    print("\nLLM Response:")
    print(format_prediction_output(action, reasoning))

if __name__ == "__main__":
    main() 