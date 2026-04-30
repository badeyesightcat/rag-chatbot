"""
Phase 0: Intent Classification
Goal:   Inspect the user message → assign one of 4 intents
Input:  raw user message string
Output: dict { intent, confidence, reason, should_rag, direct_reply }
"""

import json
import re
from enum import Enum

from app.llm.base import get_llm
from app.pipeline.observer import log_phase


class Intent(str, Enum):
    RAG_QUERY         = "RAG_QUERY"          # needs document retrieval
    CHITCHAT          = "CHITCHAT"           # greeting / small talk
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"  # world knowledge, no docs needed
    OUT_OF_SCOPE      = "OUT_OF_SCOPE"       # polite refusal


# OUT_OF_SCOPE gets a canned reply — zero token cost
CANNED_REPLIES: dict[Intent, str] = {
    Intent.OUT_OF_SCOPE: (
        "I'm a document assistant. "
        "I can only answer questions about uploaded documents "
        "or general knowledge topics."
    ),
}

CLASSIFIER_PROMPT = """You are an intent classifier for a document-based RAG chatbot.

Classify the user message into EXACTLY ONE of these intents:
- RAG_QUERY         : needs information from uploaded documents to answer
- CHITCHAT          : greeting, thanks, small talk, or casual conversation
- GENERAL_KNOWLEDGE : answerable from world knowledge, no documents needed
- OUT_OF_SCOPE      : unrelated requests (creative tasks, weather, jokes, etc.)

Respond with ONLY a JSON object, no other text:
{{
  "intent": "<one of the four intents above>",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence explaining your decision>"
}}

User message: {message}"""


class IntentClassifier:

    def __init__(self):
        self.llm = get_llm()

    def classify(self, message: str) -> dict:
        """
        Returns:
            intent       : Intent enum value as string
            confidence   : float 0–1
            reason       : short explanation (shown in admin traces)
            should_rag   : bool — the routing flag rag_agent.py acts on
            direct_reply : str | None — pre-built reply for non-RAG intents
        """
        with log_phase("classification", message=message[:80]):
            raw    = self.llm.generate(CLASSIFIER_PROMPT.format(message=message))
            parsed = self._parse(raw)

        intent = Intent(parsed["intent"])

        result: dict = {
            "intent":       intent.value,
            "confidence":   parsed.get("confidence", 1.0),
            "reason":       parsed.get("reason", ""),
            "should_rag":   intent == Intent.RAG_QUERY,
            "direct_reply": None,
        }

        # Build the direct reply for non-RAG intents
        if intent == Intent.CHITCHAT:
            result["direct_reply"] = self.llm.generate(
                f"Reply briefly and friendly to: {message}"
            )
        elif intent == Intent.GENERAL_KNOWLEDGE:
            result["direct_reply"] = self.llm.generate(
                f"Answer this general knowledge question concisely: {message}"
            )
        elif intent in CANNED_REPLIES:
            result["direct_reply"] = CANNED_REPLIES[intent]   # zero LLM cost

        return result

    def _parse(self, raw: str) -> dict:
        """
        Safely extract JSON from the LLM response.
        LLMs sometimes wrap JSON in markdown fences — strip those first.
        Falls back to RAG_QUERY if parsing fails (safe default).
        """
        try:
            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE
            )
            return json.loads(cleaned)
        except (json.JSONDecodeError, KeyError):
            return {
                "intent":     "RAG_QUERY",
                "confidence": 0.5,
                "reason":     "parse error — defaulting to RAG",
            }
