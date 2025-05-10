# Financial AI Analyst

An AI-powered stock analysis tool that provides trading recommendations based on historical data up to June 2022. The system uses GPT-4 to analyze various financial indicators and generate detailed trading recommendations.

## Features

- Technical Analysis
  - RSI (Relative Strength Index)
  - MACD
  - Bollinger Bands
  - Volume Analysis
- Financial Ratios
  - Debt-to-Equity
  - Return on Equity (ROE)
  - Return on Assets (ROA)
  - Beta
  - Dividend Yield
- Market Data
  - Historical Price Data
  - Earnings Reports
  - Analyst Ratings
  - Institutional Ownership
  - Short Interest

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd stocker
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the main script:
```bash
python src/main.py
```

Enter a stock ticker when prompted (e.g., AAPL, GOOGL, MSFT).

The system will:
1. Fetch historical data up to June 2022
2. Calculate technical indicators
3. Gather financial ratios and market data
4. Generate a trading recommendation using GPT-4

## Output Format

The system provides:
- Trading Action (Buy/Sell/Hold/Short)
- Entry Price
- Target Price
- Stop-Loss Price
- Detailed Reasoning

## Data Cutoff

All analysis is based on data up to June 2022 to ensure fair evaluation of the system's performance.

## Project Structure

```
stocker/
├── src/
│   ├── ingestion/
│   │   └── data_fetcher.py
│   ├── gpt_agent/
│   │   └── analyst.py
│   └── main.py
├── data/
├── requirements.txt
└── README.md
```

## Dependencies

- yfinance: Stock data fetching
- pandas: Data manipulation
- ta: Technical analysis
- openai: GPT-4 integration
- python-dotenv: Environment variable management

## License

MIT License 