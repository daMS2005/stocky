import os

REQUIRED_KEYS = [
    'FMP_API_KEY',
    'FRED_API_KEY',
    'OPENAI_API_KEY',
    'TWITTER_BEARER_TOKEN',
    'TWITTER_API_KEY',
    'TWITTER_API_SECRET',
    'REDDIT_API_KEY',
    'LINKEDIN_ACCESS_TOKEN',
]

def test_api_keys():
    print("API Key Status:")
    for key in REQUIRED_KEYS:
        value = os.getenv(key)
        if value:
            print(f"{key}: FOUND")
        else:
            print(f"{key}: MISSING")

if __name__ == "__main__":
    test_api_keys() 