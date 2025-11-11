from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class AgentResult:
    answer: str
    sources: List[Dict[str, Any]]

class BaseAgent:
    def run(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        raise NotImplementedError
