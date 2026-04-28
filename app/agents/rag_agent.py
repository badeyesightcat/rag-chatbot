# LangGraph workflow — 5 nodes, each = one RAG phase
#
# Flow with classification:
#
#   classify → should_rag? ──Yes──► retrieve → generate → evaluate
#                          └──No──► direct_respond → END
#
# The conditional edge is the key learning here:
# LangGraph lets you branch the graph based on state values,
# just like an if/else — but visualisable and traceable.

from langgraph.graph import StateGraph, END
from typing import TypedDict
from app.pipeline.classifier import IntentClassifier
from app.pipeline.retriever import Retriever
from app.pipeline.generator import Generator
from app.pipeline.evaluator import Evaluator
from app.config import settings

classifier = IntentClassifier()
retriever = Retriever()
generator = Generator()
evaluator = Evaluator()

class RAGState(TypedDict):
    question: str
    intent: str # filled by classify node - visible in API response
    confidence: float # classifier confidence score
    should_rag: bool # True = run retrieval; False = short-circuit
    direct_reply: str | None # pre-build reply for non-RAG intents
    chunks: list # filled by retrieve node
    generation: dict  # filled by generate node
    eval_scores: dict # filled by evaluate nod

# ---- NODE DEFINITIONS ----
def classify_node(state: RAGState) -> RAGState:
    """Phase 0: classify intent, decide routing."""
    result = classifier.classify(state["question"])
    return {
        **state,
        "intent": result["intent"],
        "confidence": result["confidence"],
        "should_rag": result["should_rag"],
        "direct_reply": result["direct_reply"]
    }

def direct_respond_node(state: RAGState) -> RAGState:
    """Short-circuit path: wrap direct_reply in generation format for uniform API output."""
    return {
        **state,
        "chunks": [],
        "generation": {
            "answer": state["direct_reply"],
            "prompt": "[no retrieval - direct response]",
            "sources": []
        },
        "eval_scores": {} # RAGAS skipped for non-RAG responses
    }

def retrieve_node(state: RAGState) -> RAGState:
    chunks = retriever.retrieve(state["question"])
    return {**state, "chunks": chunks}

def generate_node(state: RAGState) -> RAGState:
    gen = generator.generate(state["question"], state["chunks"])
    return {**state, "generation": gen}

def evaluate_node(state: RAGState) -> RAGState:
    if not settings.EVAL_ENABLED:
        return { **state, "eval_scores": {} }
    scores = evaluator.score(
        state["question"],
        state["generation"]["answer"],
        state["chunks"]
    )
    return {**state, "eval_scores": scores}

# ---- ROUTING FUNCTION ----
# This is what makes LangGraph different from a plain function chain.
# After classify_node runs, the graph calls this function to decide
# which node to execute NEXT — dynamic branching based on state.

def route_after_classify(state: RAGState) -> str:
    """Returns the name of the next node to run."""
    if state["should_rag"]:
        return "retrieve" # full pipeline
    else:
        return "direct_respond" #short-circuit

# ---- BUILD THE GRAPH ----
graph = StateGraph(RAGState)

graph.add_node("classify", classify_node)
graph.add_node("direct_respond", direct_respond_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("evaluate", evaluate_node)

graph.set_entry_point("classify") # always starts here now

# Conditional edge: classify → (retrieve OR direct_respond) based on route_after_classify
graph.add_conditional_edge("classify", route_after_classify)

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "evaluate")
graph.add_edge("evaluate", END)
graph.add_node("direct_respond", END) # short-circuit path ends here

rag_pipeline = graph.compile()

def run_rag(question: str) -> RAGState:
    return rag_pipeline.invoke({
        "question": question,
        "intent": "",
        "confidence": 0.0,
        "should_rag": False,
        "direct_reply": None,
        "chunks": [],
        "generation": {},
        "eval_scores": {}
        })