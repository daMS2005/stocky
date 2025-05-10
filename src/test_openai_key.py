import os
from openai import OpenAI, OpenAIError

def test_openai_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY: MISSING")
        return

    try:
        client = OpenAI(api_key=api_key)
        # Make a simple call to list models (does not use tokens)
        models = client.models.list()
        print("OPENAI_API_KEY: VALID ✅")
        print(f"Number of models available: {len(models.data)}")
        # Check if the model 'gpt-4o' is available
        model_name = 'gpt-4o'
        if any(model.id == model_name for model in models.data):
            print(f"Model {model_name} is available.")
        else:
            print(f"Model {model_name} is not available.")
    except OpenAIError as e:
        print("OPENAI_API_KEY: INVALID ❌")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_openai_key() 