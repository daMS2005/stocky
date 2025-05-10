# import pytest
from src.gpt_agent.analyst import GPTAnalyst

class DummyClient:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                class DummyResponse:
                    class choices:
                        message = type('obj', (object,), {'content': 'not a json'})
                    choices = [choices()]
                return DummyResponse()

def test_gptanalyst_invalid_json(monkeypatch):
    analyst = GPTAnalyst()
    # Patch the client to use the dummy
    analyst.client = DummyClient()
    features = {}
    ticker = 'AAPL'
    result = analyst.get_recommendation(features, ticker)
    assert result['action'] == 'ERROR'
    assert result['entry_price'] is None
    assert result['target_price'] is None
    assert result['stop_loss'] is None
    assert 'Failed to parse GPT response' in result['reasoning']

if __name__ == "__main__":
    test_gptanalyst_invalid_json(None)
