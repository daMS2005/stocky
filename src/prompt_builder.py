from src.rag_db import get_vector_store
import openai
import numpy as np

def build_prediction_prompt(ticker, current_week_start, current_features, current_texts, num_similar=3):
    """
    Build a prompt for the LLM by retrieving similar past weeks from the RAG DB and formatting them along with the current week's data.
    
    Args:
        ticker (str): The stock ticker (e.g., 'AAPL').
        current_week_start (date): The start date of the current week.
        current_features (dict): The features for the current week (e.g., RSI, PE ratio, open/close prices).
        current_texts (list): The texts for the current week (e.g., news headlines, earnings updates).
        num_similar (int): Number of similar past weeks to retrieve (default: 3).
    
    Returns:
        str: A formatted prompt for the LLM.
    """
    # 1. Get the vector store
    vector_store = get_vector_store('faiss', embedding_dim=1536)
    
    # 2. Embed the current week's texts
    current_embeddings = []
    for text in current_texts:
        response = openai.embeddings.create(model="text-embedding-3-small", input=text)
        emb = response.data[0].embedding
        current_embeddings.append(np.array(emb, dtype='float32'))
    
    # 3. Query the RAG DB for similar past weeks
    similar_records = vector_store.query_similar(current_embeddings[0].tolist(), num_similar)
    
    # 4. Format the prompt
    prompt = f"Current Week ({ticker}, Week of {current_week_start}):\n"
    prompt += f"Features: {current_features}\n"
    prompt += f"Texts: {current_texts}\n\n"
    
    prompt += "Similar Past Weeks:\n"
    for i, rec in enumerate(similar_records, 1):
        price_data = rec.get('price_data', {})
        open_price = price_data.get('open_price', 'N/A')
        close_price = price_data.get('close_price', 'N/A')
        weekly_return = price_data.get('weekly_return', 'N/A')
        
        prompt += f"{i}. Week of {rec['week_start']}:\n"
        prompt += f"   Features: {rec['features']}\n"
        prompt += f"   Texts: {rec['texts']}\n"
        prompt += f"   Price Movement: Open ${open_price:.2f} → Close ${close_price:.2f} ({weekly_return:+.2f}%)\n\n"
    
    prompt += f"Instruction: Based on the current week's data and these similar past weeks, predict the likely movement for {ticker} this week. Consider the actual price movements in similar past weeks. Provide reasoning and a recommendation (Buy, Hold, Sell)."
    
    return prompt 