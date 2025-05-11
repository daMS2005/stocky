import os
from typing import Dict, Any
import json
from openai import OpenAI
import datetime
import re

class GPTAnalyst:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def _construct_prompt(self, features: Dict[str, Any], ticker: str) -> str:
        """Construct a detailed prompt for GPT-4 with all available features."""
        # Get the latest price data
        price_data = features.get('price_data', [])
        if not price_data:
            print(f"Warning: price_data is empty for {ticker} in _construct_prompt.")
            current_price = 'N/A'
        else:
            latest_price = price_data[-1]
            current_price = latest_price.get('Close')
            if current_price is None:
                current_price = 'N/A'
            else:
                try:
                    current_price = f"{float(current_price):.2f}"
                except Exception:
                    current_price = str(current_price)
        last_updated = features.get('last_updated')
        if last_updated is None:
            last_updated = 'N/A'
        technical_signals = json.dumps(features.get('technical_signals', {}) or {}, indent=2)
        sentiment = json.dumps(features.get('sentiment', {}) or {}, indent=2)
        financial_ratios = json.dumps(features.get('fundamental', {}).get('financial_ratios', {}) or {}, indent=2)
        earnings = json.dumps(features.get('fundamental', {}).get('earnings', {}) or {}, indent=2)
        macro_indicators = json.dumps(features.get('fundamental', {}).get('macro_indicators', {}) or {}, indent=2)
        
        # Extract market features (prefer new dict, fallback to old keys)
        market_features = features.get('market_features', {})
        fx_rates = json.dumps(
            market_features.get('fx_rates') or features.get('fx_rates', {}) or {}, indent=2
        )
        country_risk = (
            market_features.get('country_risk')
            if 'country_risk' in market_features else features.get('country_risk', 'N/A')
        )
        commodity_prices = json.dumps(
            market_features.get('commodity_prices') or features.get('commodity_prices', {}) or {}, indent=2
        )
        company_commodities = ', '.join(
            market_features.get('company_commodities') or features.get('company_commodities', [])
        ) or 'None'
        vix = (
            market_features.get('vix') if 'vix' in market_features else features.get('vix', 'N/A')
        )
        garch_volatility = (
            market_features.get('garch_volatility') if 'garch_volatility' in market_features else features.get('garch_volatility', 'N/A')
        )
        
        # Lagged features summary
        lagged_features = []
        for lag in range(1, 4):
            close = price_data[-1].get(f'Close_t-{lag}', 'N/A') if price_data else 'N/A'
            ret = price_data[-1].get(f'Return_t-{lag}', 'N/A') if price_data else 'N/A'
            lagged_features.append(f"Close_t-{lag}: {close}, Return_t-{lag}: {ret}")
        lagged_features_summary = '\n'.join(lagged_features)

        prompt = f"""You are an expert financial analyst. Analyze the following data for {ticker} and provide a detailed trading recommendation.
        
Current Price: ${current_price}
Last Updated: {last_updated}

Technical Analysis:
{technical_signals}

Sentiment Analysis:
{sentiment}

Fundamental Analysis:
Financial Ratios:
{financial_ratios}

Earnings Data:
{earnings}

Macroeconomic Indicators:
{macro_indicators}

Market & Macro Features:
FX Rates:
{fx_rates}

Country Risk Index:
{country_risk}

Relevant Commodities for {ticker}:
{company_commodities}

Commodity Prices:
{commodity_prices}

Volatility Indices:
- VIX: {vix}
- GARCH Volatility: {garch_volatility}

Lagged Features (recent days):
{lagged_features_summary}

Based on this comprehensive data, provide a trading recommendation in the following JSON format. **Return only strict, valid JSON. Do not include comments, duplicate fields, or trailing commas. All required fields must be present.**

Example:
{{
  "prediction": {{
    "action": "Buy",
    "entry_price": 134.5,
    "target_price": 141.0,
    "stop_loss": 131.0,
    "expected_return_pct": 4.83,
    "confidence": 0.85,
    "reasoning": "Strong technicals, insider buying, and past analog support this strategy."
  }},
  "used_features": ["RSI", "MACD", "Volume_MA", "sentiment_positive"],
  "prompt_feedback": "[AAPL] 2022-07-01–2022-07-07: GPT strategy based on insider sentiment + RSI"
}}

Return only the JSON object, nothing else. Do not use numbered keys in arrays. Do not duplicate fields. All fields must be present, even if empty or null.

Provide a clear, data-driven explanation for your recommendation that considers all available factors."""

        return prompt
    
    def get_recommendation(self, features: Dict[str, Any], ticker: str, timeframe: dict = None) -> Dict[str, Any]:
        """Get trading recommendation from GPT-4."""
        prompt = self._construct_prompt(features, ticker)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert financial analyst providing trading recommendations based on historical data up to June 2022. Consider technical, fundamental, and sentiment analysis in your decision."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        raw_content = response.choices[0].message.content
        cleaned_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_content.strip(), flags=re.IGNORECASE)
        try:
            gpt_response = json.loads(cleaned_content)
            if not isinstance(gpt_response, dict):
                print(f"LLM response is not a dict: {type(gpt_response)} - {gpt_response}")
                gpt_response = {}
            prediction = gpt_response.get("prediction", {})
            if not isinstance(prediction, dict):
                print(f"Prediction is not a dict: {type(prediction)} - {prediction}")
                prediction = {}
            used_features = gpt_response.get("used_features", [])
            prompt_feedback = gpt_response.get("prompt_feedback", "")
            # Ensure all numeric fields are properly handled
            numeric_fields = ['entry_price', 'target_price', 'stop_loss', 'confidence', 'expected_return_pct']
            for field in numeric_fields:
                value = prediction.get(field)
                if value is None or value == 'N/A':
                    prediction[field] = None
                else:
                    try:
                        prediction[field] = round(float(value), 2)
                    except (ValueError, TypeError):
                        prediction[field] = None
            # Ensure action is a valid string
            if not isinstance(prediction.get('action'), str):
                prediction['action'] = 'ERROR'
            # Ensure reasoning is a valid string
            if not isinstance(prediction.get('reasoning'), str):
                prediction['reasoning'] = 'Invalid reasoning provided'
            # Inject ticker, date, timeframe from app logic
            output = {
                "ticker": ticker,
                "date": datetime.date.today().isoformat(),
                "timeframe": timeframe or {},
                "prediction": prediction,
                "used_features": used_features,
                "prompt_feedback": prompt_feedback
            }
            return output
        except Exception as e:
            print(f"Failed to parse GPT response: {e}\nRaw response: {raw_content}")
            return {
                "ticker": ticker,
                "date": datetime.date.today().isoformat(),
                "timeframe": timeframe or {},
                "prediction": {
                    "action": "ERROR",
                    "entry_price": None,
                    "target_price": None,
                    "stop_loss": None,
                    "confidence": None,
                    "expected_return_pct": None,
                    "reasoning": f"Failed to parse GPT response: {e}\nRaw response: {raw_content}"
                },
                "used_features": [],
                "prompt_feedback": ""
            }

    def get_recommendation_with_feedback(self, features: Dict[str, Any], ticker: str, prompt_feedback: str, timeframe: dict = None) -> Dict[str, Any]:
        """Get trading recommendation from GPT-4, including feedback context in the prompt."""
        base_prompt = self._construct_prompt(features, ticker)
        prompt = f"""{base_prompt}\n\n---\n\nFeedback from similar past cases:\n{prompt_feedback}\n\nPlease consider this feedback when making your recommendation."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert financial analyst providing trading recommendations based on historical data up to June 2022. Consider technical, fundamental, and sentiment analysis in your decision."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        raw_content = response.choices[0].message.content
        cleaned_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_content.strip(), flags=re.IGNORECASE)
        try:
            gpt_response = json.loads(cleaned_content)
            # Only extract prediction, used_features, prompt_feedback
            prediction = gpt_response.get("prediction", {})
            used_features = gpt_response.get("used_features", [])
            prompt_feedback = gpt_response.get("prompt_feedback", "")
            # Ensure all numeric fields are properly handled
            numeric_fields = ['entry_price', 'target_price', 'stop_loss', 'confidence', 'expected_return_pct']
            for field in numeric_fields:
                value = prediction.get(field)
                if value is None or value == 'N/A':
                    prediction[field] = None
                else:
                    try:
                        prediction[field] = round(float(value), 2)
                    except (ValueError, TypeError):
                        prediction[field] = None
            # Ensure action is a valid string
            if not isinstance(prediction.get('action'), str):
                prediction['action'] = 'ERROR'
            # Ensure reasoning is a valid string
            if not isinstance(prediction.get('reasoning'), str):
                prediction['reasoning'] = 'Invalid reasoning provided'
            # Inject ticker, date, timeframe from app logic
            output = {
                "ticker": ticker,
                "date": datetime.date.today().isoformat(),
                "timeframe": timeframe or {},
                "prediction": prediction,
                "used_features": used_features,
                "prompt_feedback": prompt_feedback
            }
            return output
        except Exception as e:
            print(f"Failed to parse GPT response: {e}\nRaw response: {raw_content}")
            return {
                "ticker": ticker,
                "date": datetime.date.today().isoformat(),
                "timeframe": timeframe or {},
                "prediction": {
                    "action": "ERROR",
                    "entry_price": None,
                    "target_price": None,
                    "stop_loss": None,
                    "confidence": None,
                    "expected_return_pct": None,
                    "reasoning": f"Failed to parse GPT response: {e}\nRaw response: {raw_content}"
                },
                "used_features": [],
                "prompt_feedback": ""
            } 