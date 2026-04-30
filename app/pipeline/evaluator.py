"""
Phase 5: Evaluation
Run RAGAS scoring after every RAG_QUERY response.
Scores are stored in SQLite and plotted over time in the admin dashboard.

Three metrics:
  faithfulness      — are all claims in the answer supported by the context?
  answer_relevancy  — does the answer address the question?
  context_precision — were the retrieved chunks actually relevant?
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from app.db import save_eval_result
from app.pipeline.observer import log_phase


class Evaluator:

    def score(self, question: str, answer: str, contexts: list[dict]) -> dict:
        """
        Runs RAGAS evaluation and persists results.
        Returns scores dict (empty dict if evaluation is disabled or fails).

        Note: RAGAS internally calls an LLM, so this adds latency.
        Set EVAL_ENABLED=false in .env to skip during fast dev iteration.
        """
        dataset = Dataset.from_dict({
            "question": [question],
            "answer":   [answer],
            "contexts": [[c["text"] for c in contexts]],
        })

        with log_phase("evaluation"):
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
            )

        scores = {
            "faithfulness":     round(float(result["faithfulness"]),     3),
            "answer_relevancy": round(float(result["answer_relevancy"]), 3),
            "context_precision":round(float(result["context_precision"]),3),
        }

        save_eval_result(question=question, answer=answer, scores=scores)
        return scores
