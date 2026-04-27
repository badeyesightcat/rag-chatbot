# =============================================================
# PHASE 5: EVALUATION
# RAGAS scores are computed after every query and stored in DB.
# The admin dashboard plots these over time so you can see
# how your parameter changes affected quality.
# =============================================================

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from app.db import save_eval_result
from app.pipeline.observer import log_phase
import asyncio

class Evaluator:

    def score(self, question: str, answer: str, contexts: list) -> dict:
        """
        Run after every query. Stores scores in DB.
        Note: RAGAS uses an LLM internally — it may be slow.
        Set EVAL_ENABLED=false in .env to disable for dev.
        """
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [[c["text"] for c in contexts]]
        })

        with log_phase("evaluation"):
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision]
            )
        
        scores = {
            "faithfulness": round(result["faithfulness"], 3),
            "answer_relevancy": round(result["answer_relevancy"], 3),
            "context_precision": round(result["context_precision"], 3)
        }

        # Persist — plotted in admin dashboard over time
        save_eval_result(question=question, answer=answer, scores=scores)
        return scores