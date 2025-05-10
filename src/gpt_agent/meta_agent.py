import os
from typing import Dict, Any, List
from datetime import datetime
from .analyst import GPTAnalyst
from src.feedback.logger import AgentLogger
from src.reporting.financial_report import generate_structured_json, generate_natural_language_report
import json

class MetaAgentController:
    def __init__(self):
        self.analyst = GPTAnalyst()
        self.logger = AgentLogger()
        self.feedback_dict = {}  # In-memory feedback log deduplication

    def make_prediction(self, ticker: str, timeframe: Dict[str, str], features: Dict[str, Any]) -> Dict[str, Any]:
        # Retrieve similar past cases (RAG feedback)
        similar_cases = self.logger.embed_and_retrieve_similar(features)
        feedback_summary = self._summarize_feedback(similar_cases)

        # Construct prompt with feedback
        prompt_feedback = feedback_summary or "No similar past cases found."
        # Get recommendation from analyst
        prediction_output = self.analyst.get_recommendation_with_feedback(features, ticker, prompt_feedback, timeframe)

        # Compose reasoning and used features
        reasoning = prediction_output["prediction"].get('reasoning', '')
        used_features = prediction_output.get('used_features', list(features.keys()))
        generated_date = prediction_output.get('date', features.get('last_updated', 'N/A'))
        prompt_feedback = prediction_output.get('prompt_feedback', prompt_feedback)

        # Extract current price from features
        current_price = None
        if features.get('price_data') and len(features['price_data']) > 0:
            current_price = features['price_data'][-1].get('Close')

        # Prepare structured output
        structured = generate_structured_json(
            ticker=ticker,
            timeframe=timeframe,
            prediction=prediction_output["prediction"],
            reasoning=reasoning,
            used_features=used_features,
            prompt_feedback=prompt_feedback,
            date=generated_date,
            current_price=current_price
        )
        # Prepare natural language report
        report = generate_natural_language_report(
            ticker=ticker,
            timeframe=timeframe,
            prediction=prediction_output["prediction"],
            reasoning=reasoning,
            used_features=used_features,
            prompt_feedback=prompt_feedback,
            current_price=current_price
        )
        # Deduplicate feedback logs for this (ticker, timeframe)
        self._log_feedback_dedup(ticker, timeframe, structured, report, features, generated_date)
        return {"structured": structured, "report": report, "feedback": prompt_feedback}

    def _summarize_feedback(self, cases: List[Dict[str, Any]]) -> str:
        if not cases:
            return ""
        summary = []
        for case in cases:
            tf = case.get('timeframe', {})
            summary.append(f"[{case.get('ticker')}] {tf.get('start')}–{tf.get('end')}: {case.get('prediction', {}).get('action')} | {case.get('reasoning', '')[:100]}...")
        return "\n".join(summary)

    def evaluate_and_log_result(self, ticker: str, timeframe: Dict[str, str], actual_outcome: Dict[str, Any]):
        # Log the actual outcome for the period
        self.logger.log_entry({
            "ticker": ticker,
            "timeframe": timeframe,
            "actual_outcome": actual_outcome,
            "type": "evaluation",
            "logged_at": datetime.utcnow().isoformat()
        })

    def adapt_strategy(self):
        # Placeholder: In production, analyze logs and adjust feature selection/weights if repeated mistakes
        pass

    def _log_feedback_dedup(self, ticker, timeframe, structured, report, features, generated_date):
        key = f"{ticker}-{timeframe['start']}-{timeframe['end']}"
        self.feedback_dict[key] = {
            **structured,
            "report": report,
            "features": features,
            "type": "prediction",
            "date": generated_date
        }
        # Overwrite the log file with deduplicated feedback
        logs = self.logger.get_all_logs()
        # Remove any previous log for this key
        new_logs = [log for log in logs if not (log.get('ticker') == ticker and log.get('timeframe', {}) == timeframe)]
        # Add the new deduped log
        new_logs.append(self.feedback_dict[key])
        with open(self.logger.log_file, 'w') as f:
            for log in new_logs:
                f.write(json.dumps(log) + '\n') 