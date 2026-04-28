# =============================================================
# PHASE 4: GENERATION
# The prompt template is the most impactful thing you can tune.
# This template:
#   - Grounds the LLM in the retrieved context
#   - Forces citation of sources
#   - Prevents hallucination with explicit instruction
# =============================================================

from app.llm.base import get_llm
from app.pipeline.observer import log_phase

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY
the context below. If the context does not contain the answer, say
"I don't have enough information to answer this."

Do not make up information. Cite the source document when possible.

=== Context ===
{context}

=== Question ===
{question}

=== Answer ==="""

class Generator:

    def __init__(self):
        self.llm = get_llm()

    def generate(self, question: str, retrieved_chunks: list[dict]) -> dict:
        # Format context: numbered list with source attribution
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source_label = f"[{i}] {chunk['source']} p.{chunk['page']}"
            context_parts.append(f"{source_label}\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        with log_phase("generation", prompt_len=len(prompt)):
            answer = self.llm.generate(prompt)
        
        return {
            "answer": answer,
            "prompt": prompt, # visible in admin UI → debug prompt
            "sources": [{
                "text": c["text"][:200],
                "source": c["source"],
                "page": c["page"],
                "score": c["rerank_score"]
            } for c in retrieved_chunks]
        }