import unittest
from src.gpt_agent.analyst import GPTAnalyst

class TestGPTAnalyst(unittest.TestCase):
    def test_construct_prompt_with_none_values(self):
        analyst = GPTAnalyst()
        features = {
            'price_data': [{'Close': None}],
            'last_updated': None,
            'technical_signals': None,
            'sentiment': None,
            'fundamental': {
                'financial_ratios': None,
                'earnings': None,
                'macro_indicators': None
            }
        }
        ticker = 'AAPL'
        prompt = analyst._construct_prompt(features, ticker)
        self.assertIn('N/A', prompt)
        self.assertIn('{}', prompt)

if __name__ == '__main__':
    unittest.main() 