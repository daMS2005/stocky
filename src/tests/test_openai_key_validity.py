import os
from openai import OpenAI

API_KEY = "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"

def test_openai_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        response = client.models.list()
        print("✅ OpenAI API key is valid!")
        return True
    except Exception as e:
        print(f"❌ OpenAI API key is INVALID: {e}")
        return False

if __name__ == "__main__":
    test_openai_key(API_KEY) 