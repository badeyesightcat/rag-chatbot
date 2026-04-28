# LangGraph workflow — 4 nodes, each = one RAG phase
# State flows: retrieve → generate → evaluate → respond
# No Kafka needed — async Python handles concurrency fine for a chatbot
from langgraph.graph import StateGraph, END
from typing import TypedDict
from app.pipeline.retriever import Retriever
from app.pipeline.generator import Generator
from app.pipeline.evaluator import Evaluator
from app.config import settings

retriever = Retriever()
generator = Generator()
evaluator = Evaluator()

class RAGState(TypedDict):
    question: str
    chunks: list # filled by retrieve node
    generation: dict  # filled by generate node
    eval_scores: dict # filled by evaluate nod

def retrieve_node(state: RAGState) -> RAGState:
    chunks = retriever.retrieve(state["question"])
    return {**state, "chunks": chunks}

def generate_node(state: RAGState) -> RAGState:
    gen = generator.generate(state["question"], state["chunks"])
    return {**state, "generation": gen}

def evaluate_node(state: RAGState) -> RAGState:
    if not settings.EVAL_ENABLED:
        return { **state, "eval_scores": {}}
    scores = evaluator.score(
        state["question"],
        state["generation"]["answer"],
        state["chunks"]
        )
    return {**state, "eval_scores": scores}

# Build the graph
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("evaluate", evaluate_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "evaluate")
graph.add_edge("evaluate", END)
rag_pipeline = graph.compile()

def run_rag(question: str) -> RAGState:
    return rag_pipeline.invoke({
        "question": question,
        "chunks": [],
        "generation": {},
        "eval_scores": {}
        })