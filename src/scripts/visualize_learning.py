import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List
import numpy as np

def load_metrics(metrics_file: str) -> pd.DataFrame:
    """Load and process metrics data."""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics)
    df['date'] = pd.to_datetime(df['date'])
    return df

def plot_accuracy_trend(df: pd.DataFrame, save_path: str):
    """Plot accuracy trends over time."""
    plt.figure(figsize=(12, 6))
    
    # Plot overall accuracy
    plt.plot(df['date'], df['overall_accuracy'], 
             label='Overall Accuracy', linewidth=2)
    
    # Plot rolling accuracy
    plt.plot(df['date'], df['rolling_accuracy'], 
             label='Rolling Accuracy (5 weeks)', linewidth=2, linestyle='--')
    
    # Add trend line
    z = np.polyfit(range(len(df)), df['overall_accuracy'], 1)
    p = np.poly1d(z)
    plt.plot(df['date'], p(range(len(df))), 
             label='Trend Line', color='red', linestyle=':')
    
    plt.title('Prediction Accuracy Over Time')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'accuracy_trend.png'))
    plt.close()

def plot_returns_distribution(df: pd.DataFrame, save_path: str):
    """Plot distribution of returns."""
    plt.figure(figsize=(10, 6))
    
    # Create histogram with KDE
    sns.histplot(data=df, x='average_return', bins=30, kde=True)
    
    plt.title('Distribution of Weekly Returns')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add mean and median lines
    mean_return = df['average_return'].mean()
    median_return = df['average_return'].median()
    
    plt.axvline(mean_return, color='red', linestyle='--', 
                label=f'Mean: {mean_return:.2f}%')
    plt.axvline(median_return, color='green', linestyle='--', 
                label=f'Median: {median_return:.2f}%')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'returns_distribution.png'))
    plt.close()

def plot_prediction_patterns(logger, save_path: str):
    """Plot patterns in predictions and their outcomes."""
    # Get all predictions
    predictions = logger.get_past_predictions('AAPL', n=1000)  # Get all predictions
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    
    # Create pivot table for prediction types
    pred_counts = pd.crosstab(pred_df['prediction'], pred_df['outcome'])
    
    # Plot prediction distribution
    plt.figure(figsize=(10, 6))
    pred_counts.plot(kind='bar', stacked=True)
    plt.title('Distribution of Predictions and Their Outcomes')
    plt.xlabel('Prediction Type')
    plt.ylabel('Count')
    plt.legend(title='Outcome')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'prediction_patterns.png'))
    plt.close()
    
    # Plot confidence vs accuracy
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pred_df, x='prediction', y='confidence')
    plt.title('Confidence Levels by Prediction Type')
    plt.xlabel('Prediction Type')
    plt.ylabel('Confidence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confidence_by_prediction.png'))
    plt.close()

def plot_learning_curves(df: pd.DataFrame, save_path: str):
    """Plot learning curves showing improvement over time."""
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative accuracy
    df['cumulative_correct'] = (df['overall_accuracy'] * df['total_predictions']).cumsum()
    df['cumulative_total'] = df['total_predictions'].cumsum()
    df['cumulative_accuracy'] = df['cumulative_correct'] / df['cumulative_total']
    
    # Plot cumulative accuracy
    plt.plot(df['date'], df['cumulative_accuracy'], 
             label='Cumulative Accuracy', linewidth=2)
    
    # Plot rolling accuracy
    plt.plot(df['date'], df['rolling_accuracy'], 
             label='Rolling Accuracy (5 weeks)', linewidth=2, linestyle='--')
    
    plt.title('Learning Curves')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'learning_curves.png'))
    plt.close()

def main():
    # Configuration
    METRICS_FILE = "logs/learning_metrics.json"
    SAVE_PATH = "logs/visualizations"
    
    # Create save directory
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Load metrics
    df = load_metrics(METRICS_FILE)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Accuracy trends
    print("Plotting accuracy trends...")
    plot_accuracy_trend(df, SAVE_PATH)
    
    # 2. Returns distribution
    print("Plotting returns distribution...")
    plot_returns_distribution(df, SAVE_PATH)
    
    # 3. Learning curves
    print("Plotting learning curves...")
    plot_learning_curves(df, SAVE_PATH)
    
    # 4. Prediction patterns (if logger is available)
    try:
        from src.prediction_logger import PredictionLogger
        logger = PredictionLogger()
        print("Plotting prediction patterns...")
        plot_prediction_patterns(logger, SAVE_PATH)
    except Exception as e:
        print(f"Could not generate prediction patterns: {str(e)}")
    
    print(f"\nVisualizations saved to {SAVE_PATH}/")
    print("Generated plots:")
    print("1. accuracy_trend.png - Shows overall and rolling accuracy over time")
    print("2. returns_distribution.png - Distribution of weekly returns")
    print("3. learning_curves.png - Shows cumulative and rolling accuracy")
    print("4. prediction_patterns.png - Distribution of prediction types and outcomes")
    print("5. confidence_by_prediction.png - Confidence levels by prediction type")

if __name__ == "__main__":
    main() 