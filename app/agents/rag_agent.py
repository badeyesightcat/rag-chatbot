"""
LangGraph RAG pipeline — 5 nodes.

Flow:
  classify → should_rag? ──Yes──► retrieve → generate → evaluate → END
                         └──No──► direct_respond → END

The conditional edge is the core learning concept:
LangGraph branches the execution graph based on state values,
just like an if/else — but every branch is traceable and visualisable.
"""

from typing import TypedDict   # only TypedDict has no built-in replacement

from langgraph.graph import END, StateGraph

from app.config import settings
from app.pipeline.classifier import IntentClassifier
from app.pipeline.evaluator import Evaluator
from app.pipeline.generator import Generator
from app.pipeline.retriever import Retriever

classifier = IntentClassifier()
retriever  = Retriever()
generator  = Generator()
evaluator  = Evaluator()


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question:     str
    intent:       str           # filled by classify_node
    confidence:   float         # classifier confidence 0–1
    should_rag:   bool          # routing flag
    direct_reply: str | None    # pre-built reply for non-RAG intents
    chunks:       list          # filled by retrieve_node
    generation:   dict          # filled by generate_node
    eval_scores:  dict          # filled by evaluate_node


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_node(state: RAGState) -> RAGState:
    result = classifier.classify(state["question"])
    return {
        **state,
        "intent":       result["intent"],
        "confidence":   result["confidence"],
        "should_rag":   result["should_rag"],
        "direct_reply": result["direct_reply"],
    }


def direct_respond_node(state: RAGState) -> RAGState:
    """Short-circuit: wrap direct_reply in standard generation format."""
    return {
        **state,
        "chunks":     [],
        "generation": {
            "answer":  state["direct_reply"],
            "prompt":  "[no retrieval — direct response]",
            "sources": [],
        },
        "eval_scores": {},   # RAGAS skipped for non-RAG responses
    }


def retrieve_node(state: RAGState) -> RAGState:
    chunks = retriever.retrieve(state["question"])
    return {**state, "chunks": chunks}


def generate_node(state: RAGState) -> RAGState:
    gen = generator.generate(state["question"], state["chunks"])
    return {**state, "generation": gen}


def evaluate_node(state: RAGState) -> RAGState:
    if not settings.EVAL_ENABLED:
        return {**state, "eval_scores": {}}
    scores = evaluator.score(
        state["question"],
        state["generation"]["answer"],
        state["chunks"],
    )
    return {**state, "eval_scores": scores}


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_after_classify(state: RAGState) -> str:
    """Returns the name of the next node based on classification result."""
    return "retrieve" if state["should_rag"] else "direct_respond"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

graph = StateGraph(RAGState)

graph.add_node("classify",        classify_node)
graph.add_node("direct_respond",  direct_respond_node)
graph.add_node("retrieve",        retrieve_node)
graph.add_node("generate",        generate_node)
graph.add_node("evaluate",        evaluate_node)

graph.set_entry_point("classify")

graph.add_conditional_edges("classify", route_after_classify)

graph.add_edge("retrieve",       "generate")
graph.add_edge("generate",       "evaluate")
graph.add_edge("evaluate",       END)
graph.add_edge("direct_respond", END)

rag_pipeline = graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rag(question: str) -> RAGState:
    return rag_pipeline.invoke({
        "question":     question,
        "intent":       "",
        "confidence":   0.0,
        "should_rag":   False,
        "direct_reply": None,
        "chunks":       [],
        "generation":   {},
        "eval_scores":  {},
    })
