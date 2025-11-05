from typing import TypedDict
from langchain.chat_models import init_chat_model
from typing import Literal
from langgraph.graph import StateGraph, START, END
import langsmith as ls  # noqa: F401
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "codestral-latest",
    model_provider="mistralai",
)


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# class GraphState(MessagesState):
#    pass


def explain_node(s: GraphState):
    message = model.invoke(f"Explain the code: {s['messages'][-1].content}").content
    return {"messages": message}


def generate_node(s: GraphState):
    message = model.invoke(f"Generate code for: {s['messages'][-1].content}").content
    return {"messages": message}


def decide_request_kind(s: GraphState) -> Literal["generate_node", "explain_node"]:
    kind = model.invoke(
        f"Is this query a request to generate code, or explain code? Respond with \"explain\" or \"generate\". Query: {s['messages'][-1].content}"
    ).content
    if kind == "generate":
        return "generate_node"
    else:
        return "explain_node"


builder = StateGraph(GraphState)

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
for chunk in graph.stream({"messages": "Generate code to sort a list"}):
    print(json.dumps(chunk, indent=2))

# result = graph.invoke({"user_request": "Generate code to sort a list"})
# print(result)
