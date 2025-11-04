"""
Quick test of the interactive human-in-the-loop feature
"""

from typing import TypedDict, Literal
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import json
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "codestral-latest",
    model_provider="mistralai",
)


class RouterState(TypedDict):
    user_request: str
    result: str
    clarification_response: str


def explain_node(s: RouterState):
    return {"result": model.invoke(f"Explain the code: {s['user_request']}").content}


def generate_node(s: RouterState):
    return {"result": model.invoke(f"Generate code for: {s['user_request']}").content}


def ask_clarification_node(s: RouterState):
    return {
        "result": "I need clarification. Please say 'generate' or 'explain'.",
    }


def process_clarification_node(s: RouterState):
    clarification = s.get("clarification_response", "").lower()

    if "generate" in clarification or "1" in clarification:
        return generate_node(s)
    elif "explain" in clarification or "2" in clarification:
        return explain_node(s)
    else:
        return {"result": "Sorry, I still don't understand. Please say 'generate' or 'explain'."}


def decide_request_kind(s: RouterState) -> Literal["generate_node", "explain_node", "ask_clarification"]:
    """
    Make the router VERY conservative - if it's not explicitly clear,
    route to ask_clarification
    """

    user_request = s["user_request"].lower()

    # Very explicit generate keywords
    if any(word in user_request for word in ["write", "create", "build", "implement", "code for"]):
        return "generate_node"

    # Very explicit explain keywords
    if any(word in user_request for word in ["explain", "what is", "describe", "how does", "tell me about"]):
        return "explain_node"

    # Everything else - ask for clarification
    return "ask_clarification"


def route_after_clarification(s: RouterState) -> Literal["process_clarification", END]:
    if s.get("clarification_response"):
        return "process_clarification"
    return END


# Build graph
builder = StateGraph(RouterState)

builder.add_node("generate_node", generate_node)
builder.add_node("explain_node", explain_node)
builder.add_node("ask_clarification", ask_clarification_node)
builder.add_node("process_clarification", process_clarification_node)

builder.add_edge("generate_node", END)
builder.add_edge("explain_node", END)
builder.add_edge("process_clarification", END)

builder.add_conditional_edges(START, decide_request_kind)
builder.add_conditional_edges("ask_clarification", route_after_clarification)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["ask_clarification"])


def test_query(user_request: str, thread_id: str):
    """Test a query and handle clarification if needed"""

    print(f"\n{'='*60}")
    print(f"User: {user_request}")
    print("=" * 60)

    config = {"configurable": {"thread_id": thread_id}}

    # First run
    for chunk in graph.stream({"user_request": user_request}, config):
        print(json.dumps(chunk, indent=2))

    # Check state
    state = graph.get_state(config)

    if state.next == ("ask_clarification",):
        print("\n⚠️  I'm not sure whether you want me to:")
        print("  1. Generate new code")
        print("  2. Explain existing code or a concept")

        user_input = input("\nYour choice (1, 2, 'generate', or 'explain'): ").strip().lower()

        if user_input in ["1", "generate", "gen"]:
            clarification = "generate"
        elif user_input in ["2", "explain", "exp"]:
            clarification = "explain"
        else:
            clarification = user_input

        print(f"\n✓ Resuming with: {clarification}\n")

        # Resume
        for chunk in graph.stream({"clarification_response": clarification}, config):
            print(json.dumps(chunk, indent=2))


if __name__ == "__main__":
    print("\nTesting LangGraph Human-in-the-Loop")
    print("=" * 60)

    # Test 1: Clear generate request - should NOT ask
    test_query("Write a Python function to sort a list", "test1")

    # Test 2: Clear explain request - should NOT ask
    test_query("Explain what a decorator does in Python", "test2")

    # Test 3: Ambiguous - SHOULD ask for clarification
    test_query("recursion", "test3")
