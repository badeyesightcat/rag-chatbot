# =============================================================
# PHASE 0: INTENT CLASSIFICATION
# Goal:  Inspect the user's message → assign one of 4 intents
# Input:  raw user message string
# Output: { intent, confidence, reason, should_rag }
#
# Why LLM-based classification instead of keyword rules?
#   Rules like "if '?' in message → RAG" break constantly.
#   "Thanks for explaining machine learning!" has no '?' but
#   is clearly CHITCHAT. An LLM understands nuance.
# =============================================================

from enum import Enum
from app.llm.base import get_llm
from app.pipeline.observer import log_phase
import json, re

class Intent(str, Enum):
    RAG_QUERY = "RAG_QUERY" # needs document retrieval
    CHITCHAT = "CHITCHAT" # greeting / small talk
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE" # world knowledge, no docs needed
    OUT_OF_SCOPE = "OUT_OF_SCOPE" # refusal

# Canned replies for intents that don't touch the LLM at all
# OUT_OF_SCOPE gets a fixed reply → zero token cost
CANNED_REPLIES = {
    Intent.OUT_OF_SCOPE: "I'm a document assistant. I can only answer questions about the uploaded documents or general knowledge topics."
}

CLASSIFIER_PROMPT = """You are an intent classifier for a document-based RAG chatbot.

Classify the user message into EXACTLY ONE of these intents:
- RAG_QUERY: needs information from uploaded documents to answer
- CHITCHAT: greeting, thanks, small talk, or casual conversation
- GENERAL_KNOWLEDGE: answerable from world knowledge, no documents needed
- OUT_OF_SCOPE: unrelated requests (creative tasks, weather, jokes, etc.)

Respond with ONLY a JSON obejct, no other text:
{
    "intent": "",
    "confidence": ,
    "reason": ""
}

User message: {message}"""

class IntentClassifier:

    def __init__(self):
        self.llm = get_llm()

    def classify(self, message: str) -> dict:
        """
        Returns a dict with:
          intent      : Intent enum value
          confidence  : float 0-1
          reason      : short explanation (visible in admin UI)
          should_rag  : bool — the only field rag_agent.py needs to act on
          direct_reply: str | None — pre-built reply for non-RAG intents
        """
        with log_phase("classification", message=message[:80]):
            raw = self.llm.generate(CLASSIFIER_PROMPT.format(message=message))
            parsed = self._parse(raw)
        
        intent = Intent(parsed["intent"])

        result = {
            "intent": intent.value,
            "confidence": parsed.get("confidence", 1.0), # default to 1.0 if not provided
            "reason": parsed.get("reason", ""),
            "should_rag": intent == Intent.RAG_QUERY,
            "direct_reply": None
        }

        # For non-RAG intents, build the reply here so the agent can short-circuit
        if intent == Intent.CHITCHAT:
            result["direct_reply"] = self.llm.generate(
                f"Reply briefly and friendly to: {message}"
            )
        elif intent == Intent.GENERAL_KNOWLEDGE:
            result["direct_reply"] = self.llm.generate(
                f"Answer this general knowledge question concisely: {message}"
            )
        elif intent in CANNED_REPLIES:
            result["direct_reply"] = CANNED_REPLIES[intent] # zero llm cost

        return result
    
    def _parse(self, raw: str) -> dict:
        """
        Safely extract the JSON the LLM returned.
        LLMs sometimes wrap JSON in markdown fences — strip those first.
        Falls back to RAG_QUERY if parsing fails (safe default).
        """
        try:
            # Strip markdown code fences if present: ```json ... ```
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE)
            return json.loads(cleaned)
        except (json.JSONDecodeError, KeyError):
            # If the LLM returns garbage, default to running RAG
            return {
                "intent": "RAG_QUERY",
                "confidence": 0.5,
                "reason": "parse error - defaulting to RAG"
            }
        




