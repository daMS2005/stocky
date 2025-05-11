import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
from openai import OpenAI
import tweepy

class SentimentAnalyzer:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # Initialize Twitter client
        self.twitter_client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET')
        )
        
        # Initialize sentiment history storage
        self.sentiment_history = {}
        
    def analyze_news_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using FinBERT."""
        inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return {
            'positive': probs[0][0].item(),
            'negative': probs[0][1].item(),
            'neutral': probs[0][2].item()
        }
    
    def detect_events(self, text: str) -> List[Dict[str, Any]]:
        """Detect key events using GPT-4."""
        prompt = f"""Extract key events from the following text. For each event, provide:
1. Event type (e.g., CEO change, M&A, lawsuit, earnings)
2. Entities involved
3. Date (if mentioned)
4. Impact score (1-10)

Text: {text}

Format as JSON array of events."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at extracting key events from financial news."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        try:
            events = json.loads(response.choices[0].message.content)
            return events
        except:
            return []
    
    def get_reddit_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze Reddit sentiment for a given ticker."""
        headers = {'Authorization': f'Bearer {os.getenv("REDDIT_API_KEY")}'}
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        all_posts = []
        
        for subreddit in subreddits:
            url = f'https://oauth.reddit.com/r/{subreddit}/search?q={ticker}&restrict_sr=1&sort=relevance&t=month'
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                posts = response.json()['data']['children']
                all_posts.extend(posts)
        
        sentiments = []
        for post in all_posts:
            sentiment = self.analyze_news_sentiment(post['data']['title'] + ' ' + post['data']['selftext'])
            sentiments.append(sentiment)
        
        if sentiments:
            avg_sentiment = {
                'positive': np.mean([s['positive'] for s in sentiments]),
                'negative': np.mean([s['negative'] for s in sentiments]),
                'neutral': np.mean([s['neutral'] for s in sentiments])
            }
        else:
            avg_sentiment = {'positive': 0, 'negative': 0, 'neutral': 1}
        
        return {
            'sentiment': avg_sentiment,
            'post_count': len(all_posts),
            'source': 'reddit'
        }
    
    def get_linkedin_insights(self, ticker: str) -> Dict[str, Any]:
        """Get LinkedIn insights for a company."""
        headers = {
            'Authorization': f'Bearer {os.getenv("LINKEDIN_ACCESS_TOKEN")}',
            'X-Restli-Protocol-Version': '2.0.0'
        }
        
        # Get company updates and posts
        url = f'https://api.linkedin.com/v2/organizations/{ticker}/updates'
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            updates = response.json()
            sentiments = []
            for update in updates.get('elements', []):
                if 'commentary' in update:
                    sentiment = self.analyze_news_sentiment(update['commentary'])
                    sentiments.append(sentiment)
            
            if sentiments:
                avg_sentiment = {
                    'positive': np.mean([s['positive'] for s in sentiments]),
                    'negative': np.mean([s['negative'] for s in sentiments]),
                    'neutral': np.mean([s['neutral'] for s in sentiments])
                }
            else:
                avg_sentiment = {'positive': 0, 'negative': 0, 'neutral': 1}
            
            return {
                'sentiment': avg_sentiment,
                'update_count': len(sentiments),
                'source': 'linkedin'
            }
        
        return {
            'sentiment': {'positive': 0, 'negative': 0, 'neutral': 1},
            'update_count': 0,
            'source': 'linkedin'
        }
    
    def get_twitter_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze Twitter sentiment for a given ticker."""
        try:
            # Search for tweets about the ticker
            tweets = self.twitter_client.search_recent_tweets(
                query=f"${ticker} -is:retweet lang:en",
                tweet_fields=["public_metrics", "created_at"],
                max_results=10,  # Increased from 5 to 10 tweets per request
            )
            
            if not tweets.data:
                return {
                    'sentiment': {'positive': 0, 'negative': 0, 'neutral': 1},
                    'tweet_count': 0,
                    'source': 'twitter'
                }
            
            sentiments = []
            total_engagement = 0
            
            for tweet in tweets.data:
                # Analyze sentiment
                sentiment = self.analyze_news_sentiment(tweet.text)
                
                # Calculate engagement (likes + retweets)
                engagement = (
                    tweet.public_metrics['like_count'] +
                    tweet.public_metrics['retweet_count']
                )
                total_engagement += engagement
                
                # Weight sentiment by engagement
                weighted_sentiment = {
                    'positive': sentiment['positive'] * engagement,
                    'negative': sentiment['negative'] * engagement,
                    'neutral': sentiment['neutral'] * engagement
                }
                sentiments.append(weighted_sentiment)
            
            # Calculate weighted average sentiment
            if total_engagement > 0:
                avg_sentiment = {
                    'positive': sum(s['positive'] for s in sentiments) / total_engagement,
                    'negative': sum(s['negative'] for s in sentiments) / total_engagement,
                    'neutral': sum(s['neutral'] for s in sentiments) / total_engagement
                }
            else:
                avg_sentiment = {'positive': 0, 'negative': 0, 'neutral': 1}
            
            return {
                'sentiment': avg_sentiment,
                'tweet_count': len(tweets.data),
                'total_engagement': total_engagement,
                'source': 'twitter'
            }
            
        except Exception as e:
            print(f"Error in Twitter sentiment analysis: {str(e)}")
            return {
                'sentiment': {'positive': 0, 'negative': 0, 'neutral': 1},
                'tweet_count': 0,
                'total_engagement': 0,
                'source': 'twitter'
            }
    
    def track_sentiment_history(self, ticker: str, sentiment_data: Dict[str, Any]) -> None:
        """Track sentiment history for a ticker."""
        if ticker not in self.sentiment_history:
            self.sentiment_history[ticker] = []
            
        timestamp = datetime.now()
        self.sentiment_history[ticker].append({
            'timestamp': timestamp,
            'sentiment': sentiment_data['combined_sentiment'],
            'sources': sentiment_data['sources']
        })
        
        # Keep only last 30 days of history
        cutoff_date = timestamp - timedelta(days=30)
        self.sentiment_history[ticker] = [
            entry for entry in self.sentiment_history[ticker]
            if entry['timestamp'] > cutoff_date
        ]
    
    def calculate_sentiment_trends(self, ticker: str) -> Dict[str, Any]:
        """Calculate sentiment trends and metrics."""
        if ticker not in self.sentiment_history or not self.sentiment_history[ticker]:
            return {
                'trend': 'neutral',
                'momentum': 0,
                'volatility': 0,
                'moving_averages': {},
                'trend_strength': 0
            }
            
        history = self.sentiment_history[ticker]
        df = pd.DataFrame([
            {
                'timestamp': entry['timestamp'],
                'positive': entry['sentiment']['positive'],
                'negative': entry['sentiment']['negative'],
                'neutral': entry['sentiment']['neutral']
            }
            for entry in history
        ])
        
        # Calculate moving averages
        df['positive_ma'] = df['positive'].rolling(window=5).mean()
        df['negative_ma'] = df['negative'].rolling(window=5).mean()
        df['neutral_ma'] = df['neutral'].rolling(window=5).mean()
        
        # Calculate momentum (rate of change)
        momentum = {
            'positive': (df['positive'].iloc[-1] - df['positive'].iloc[-5]) if len(df) >= 5 else 0,
            'negative': (df['negative'].iloc[-1] - df['negative'].iloc[-5]) if len(df) >= 5 else 0,
            'neutral': (df['neutral'].iloc[-1] - df['neutral'].iloc[-5]) if len(df) >= 5 else 0
        }
        
        # Calculate volatility (standard deviation)
        volatility = {
            'positive': df['positive'].std(),
            'negative': df['negative'].std(),
            'neutral': df['neutral'].std()
        }
        
        # Determine trend direction
        if len(df) >= 5:
            positive_trend = df['positive_ma'].iloc[-1] > df['positive_ma'].iloc[-5]
            negative_trend = df['negative_ma'].iloc[-1] > df['negative_ma'].iloc[-5]
            
            if positive_trend and not negative_trend:
                trend = 'bullish'
            elif negative_trend and not positive_trend:
                trend = 'bearish'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'
            
        # Calculate trend strength (0-1)
        trend_strength = abs(
            (df['positive'].iloc[-1] - df['positive'].iloc[0]) +
            (df['negative'].iloc[-1] - df['negative'].iloc[0])
        ) / 2
        
        return {
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility,
            'moving_averages': {
                'positive': df['positive_ma'].iloc[-1] if not df['positive_ma'].empty else 0,
                'negative': df['negative_ma'].iloc[-1] if not df['negative_ma'].empty else 0,
                'neutral': df['neutral_ma'].iloc[-1] if not df['neutral_ma'].empty else 0
            },
            'trend_strength': min(trend_strength, 1.0)
        }
    
    def get_combined_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Combine sentiment from all sources and track trends."""
        reddit_sentiment = self.get_reddit_sentiment(ticker)
        linkedin_sentiment = self.get_linkedin_insights(ticker)
        twitter_sentiment = self.get_twitter_sentiment(ticker)
        
        # Weight the sources (can be adjusted)
        weights = {
            'reddit': 0.3,
            'linkedin': 0.3,
            'twitter': 0.4
        }
        
        combined_sentiment = {
            'positive': (
                reddit_sentiment['sentiment']['positive'] * weights['reddit'] +
                linkedin_sentiment['sentiment']['positive'] * weights['linkedin'] +
                twitter_sentiment['sentiment']['positive'] * weights['twitter']
            ),
            'negative': (
                reddit_sentiment['sentiment']['negative'] * weights['reddit'] +
                linkedin_sentiment['sentiment']['negative'] * weights['linkedin'] +
                twitter_sentiment['sentiment']['negative'] * weights['twitter']
            ),
            'neutral': (
                reddit_sentiment['sentiment']['neutral'] * weights['reddit'] +
                linkedin_sentiment['sentiment']['neutral'] * weights['linkedin'] +
                twitter_sentiment['sentiment']['neutral'] * weights['twitter']
            )
        }
        
        result = {
            'combined_sentiment': combined_sentiment,
            'sources': {
                'reddit': reddit_sentiment,
                'linkedin': linkedin_sentiment,
                'twitter': twitter_sentiment
            }
        }
        
        # Track sentiment history and calculate trends
        self.track_sentiment_history(ticker, result)
        result['trends'] = self.calculate_sentiment_trends(ticker)
        
        return result 

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a given text."""
        try:
            # Simple sentiment analysis using textblob
            from textblob import TextBlob
            analysis = TextBlob(text)
            return {
                'score': analysis.sentiment.polarity,
                'magnitude': abs(analysis.sentiment.polarity)
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return {'score': 0.0, 'magnitude': 0.0}

# Example replacement: Use a simple scoring mechanism
def analyze_sentiment(text):
    # Simple scoring logic (replace with your preferred non-torch approach)
    positive_words = ["good", "great", "excellent", "positive", "up", "buy", "bullish"]
    negative_words = ["bad", "poor", "negative", "down", "sell", "bearish"]
    score = 0
    for word in text.lower().split():
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    return score 