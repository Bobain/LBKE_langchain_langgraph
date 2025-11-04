from typing import TypedDict, Literal
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "codestral-latest",
    model_provider="mistralai",
)


class RouterState(TypedDict):
    user_request: str
    result: str
    user_choice: str  # User's clarification: 'generate' or 'explain'


def explain_node(s: RouterState):
    """Explain code to the user"""
    return {"result": model.invoke(f"Explain: {s['user_request']}").content}


def generate_node(s: RouterState):
    """Generate code based on user request"""
    return {"result": model.invoke(f"Generate code for: {s['user_request']}").content}


def ask_human_node(s: RouterState):
    """Placeholder - execution will pause BEFORE this node"""
    return {"result": "Waiting for clarification..."}


def decide_initial_route(s: RouterState) -> Literal["generate_node", "explain_node", "ask_human"]:
    """
    Conservative router that asks for clarification on ambiguous queries.
    Uses simple keyword matching to be predictable.
    """
    request = s["user_request"].lower()

    # Very explicit generate keywords
    if any(kw in request for kw in ["write", "create", "build", "implement", "generate", "code for", "function to"]):
        return "generate_node"

    # Very explicit explain keywords
    if any(kw in request for kw in ["explain", "what is", "what does", "describe", "how does", "tell me"]):
        return "explain_node"

    # Everything else (ambiguous) - ask for clarification
    return "ask_human"


def route_after_human(s: RouterState) -> Literal["generate_node", "explain_node"]:
    """Route based on user's clarification"""
    choice = s.get("user_choice", "").lower()
    return "generate_node" if ("generate" in choice or "1" in choice) else "explain_node"


# Build the graph
builder = StateGraph(RouterState)

# Add nodes
builder.add_node("generate_node", generate_node)
builder.add_node("explain_node", explain_node)
builder.add_node("ask_human", ask_human_node)

# Edges
builder.add_edge("generate_node", END)
builder.add_edge("explain_node", END)

# Routing
builder.add_conditional_edges(START, decide_initial_route)
builder.add_conditional_edges("ask_human", route_after_human)

# Compile with interrupt
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["ask_human"])  # Pause BEFORE asking for clarification

print(graph.get_graph().draw_ascii())


def run_with_clarification_example():
    """Example showing how to handle the interrupt and provide clarification"""

    print("\n=== Running ambiguous query ===")
    config = {"configurable": {"thread_id": "demo-1"}}

    # First invocation - will hit the interrupt
    result = graph.invoke({"user_request": "recursion"}, config)

    # Check if we need clarification
    state = graph.get_state(config)
    print(f"\n=== Current state ===")
    print(f"Next node: {state.next}")
    print(f"State values: {state.values}")

    if state.next == ("ask_human",):
        print("\n=== Graph is waiting for clarification ===")
        print("⚠️  I'm not sure whether you want me to:")
        print("  1. Generate new code")
        print("  2. Explain existing code or a concept")

        # Get actual user input
        user_input = input("\nYour choice (1, 2, 'generate', or 'explain'): ").strip().lower()

        # Parse user response
        if user_input in ["1", "generate", "gen", "create"]:
            choice = "generate"
        elif user_input in ["2", "explain", "exp"]:
            choice = "explain"
        else:
            choice = user_input

        print(f"\n=== Resuming with choice: '{choice}' ===\n")

        # Update the state with the user's choice
        graph.update_state(config, {"user_choice": choice})

        # Continue execution from where it was interrupted
        graph.invoke(None, config)

        # Get final state and show the result
        state = graph.get_state(config)
        final_result = state.values.get("result", "No result")
        print(f"=== Final Result ===")
        print(final_result)


def run_clear_request_example():
    """Example with a clear request that doesn't need clarification"""

    print("\n=== Running clear request ===")
    config = {"configurable": {"thread_id": "demo-2"}}

    result = graph.invoke({"user_request": "Generate a function to reverse a string"}, config)
    state = graph.get_state(config)
    print(f"Result: {state.values.get('result', 'No result')[:200]}...")


# Run examples
if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph with Interactive Human-in-the-Loop")
    print("=" * 60)

    run_clear_request_example()
    run_with_clarification_example()
