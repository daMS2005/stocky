import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), 'agent_logs.jsonl')

class AgentLogger:
    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                pass  # create empty file

    def log_entry(self, entry: Dict[str, Any]):
        entry['logged_at'] = datetime.utcnow().isoformat()
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_all_logs(self) -> List[Dict[str, Any]]:
        logs = []
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        return logs

    def search_logs(self, ticker: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        logs = self.get_all_logs()
        results = []
        for log in logs:
            if ticker and log.get('ticker') != ticker:
                continue
            if start_date and log.get('timeframe', {}).get('start') < start_date:
                continue
            if end_date and log.get('timeframe', {}).get('end') > end_date:
                continue
            results.append(log)
        return results

    def embed_and_retrieve_similar(self, features: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        # Placeholder: In production, use OpenAI or sentence-transformers to embed features and logs, then retrieve by similarity
        # For now, just return the most recent top_k logs
        logs = self.get_all_logs()
        return logs[-top_k:] if logs else [] 