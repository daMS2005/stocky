import openai
from typing import Dict, Any, Tuple
import json

def get_llm_prediction(prompt: str, model: str = "gpt-4o") -> Tuple[str, Dict[str, Any]]:
    """
    Send a prompt to the LLM and parse its response.
    
    Args:
        prompt: The formatted prompt string
        model: The OpenAI model to use (default: gpt-4o)
        
    Returns:
        Tuple of (action, reasoning_dict)
        - action: "BUY", "SELL", or "HOLD"
        - reasoning_dict: Dictionary containing the LLM's reasoning
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a sophisticated stock market analyst. Analyze the given data and provide a clear prediction with detailed reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for more consistent outputs
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        
        # Parse the response to extract action and reasoning
        try:
            # First try to parse as JSON
            reasoning_dict = json.loads(response_text)
            action = reasoning_dict.get("action", "HOLD")
        except json.JSONDecodeError:
            # If not JSON, try to extract action from text
            action = "HOLD"  # Default
            reasoning_dict = {"raw_response": response_text}
            
            # Look for action keywords
            if "BUY" in response_text.upper():
                action = "BUY"
            elif "SELL" in response_text.upper():
                action = "SELL"
        
        return action, reasoning_dict
        
    except Exception as e:
        print(f"Error getting LLM prediction: {str(e)}")
        return "HOLD", {"error": str(e)}

def format_prediction_output(action: str, reasoning: Dict[str, Any]) -> str:
    """
    Format the prediction output in a readable way.
    """
    output = f"\nPrediction: {action}\n"
    output += "\nReasoning:\n"
    
    if "raw_response" in reasoning:
        output += reasoning["raw_response"]
    else:
        for key, value in reasoning.items():
            if key != "action":
                output += f"{key}: {value}\n"
    
    return output 