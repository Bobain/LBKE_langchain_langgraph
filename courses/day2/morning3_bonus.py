from typing import TypedDict
from langchain.chat_models import init_chat_model
from typing import Literal
from langgraph.graph import StateGraph, START, END
import langsmith as ls  # noqa: F401
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


def explain_node(s: RouterState):
    return {"result": model.invoke(f"Explain the code: {s['user_request']}").content}


def generate_node(s):
    return {"result": model.invoke(f"Generate code for: {s['user_request']}").content}


def decide_request_kind(s: RouterState) -> Literal["generate_node", "explain_node"]:
    kind = model.invoke(
        f"Is this query a request to generate code, or explain code? Respond with \"explain\" or \"generate\". Query: {s['user_request']}"
    ).content
    if kind == "generate":
        return "generate_node"
    elif kind == "explain":
        return "explain_node"
    else:
        raise ValueError(f"Unknown kind of work: {kind}")


builder = StateGraph(RouterState)

# builder.add_node("START", START)
# builder.add_node("END", END)

# builder.set_entry_point(START)
# builder.set_finish_point(END)

builder.add_node("generate_node", generate_node)
builder.add_node("explain_node", explain_node)

builder.add_edge("generate_node", END)
builder.add_edge("explain_node", END)

builder.add_conditional_edges(START, decide_request_kind)

graph = builder.compile()

print(graph.get_graph().draw_ascii())

# Explain to me what builder.compile does in LangGraph
for chunk in graph.stream({"user_request": "Generate code to sort a list"}):
    print(json.dumps(chunk, indent=2))

# result = graph.invoke({"user_request": "Generate code to sort a list"})
# print(result)
