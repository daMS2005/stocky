from src.visualization.sentiment_visualizer import SentimentVisualizer

visualizer = SentimentVisualizer()

def test_create_sentiment_timeline_missing_date():
    # Data missing 'date' key
    sentiment_history = [
        {'sentiment': 'positive', 'score': 0.8},
        {'sentiment': 'negative', 'score': 0.2}
    ]
    try:
        fig = visualizer.create_sentiment_timeline(sentiment_history, 'AAPL')
    except KeyError as e:
        assert str(e) == "'date'"
    else:
        # If no error, check that the figure is returned
        assert fig is not None

def test_create_sentiment_heatmap_missing_date():
    sentiment_history = [
        {'sentiment': 'positive', 'score': 0.8},
        {'sentiment': 'negative', 'score': 0.2}
    ]
    try:
        fig = visualizer.create_sentiment_heatmap(sentiment_history, 'AAPL')
    except KeyError as e:
        assert str(e) == "'date'"
    else:
        assert fig is not None 

if __name__ == "__main__":
    test_create_sentiment_timeline_missing_date()
    test_create_sentiment_heatmap_missing_date() 