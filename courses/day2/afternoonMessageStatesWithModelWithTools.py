from typing import TypedDict
from langchain.chat_models import init_chat_model
from typing import Literal
from langgraph.graph import StateGraph, START, END
import langsmith as ls  # noqa: F401
import json
import importlib
from langchain.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "codestral-latest",
    model_provider="mistralai",
)


@tool
def langchain_core_exists(value: str) -> bool:
    """
    Check if a namespace or value is exported in the langchain_core module

    Example: langchain_core_exists("ChatOpenAI") # false
    langchain_core_exists(retrievers) # true
    """
    print(f"Check existence of import {value} in langchain_core")
    # https://stackoverflow.com/questions/30483246/how-can-i-check-if-a-module-has-been-imported
    found_spec = importlib.util.find_spec("." + value, package="langchain_core")
    return found_spec is not None


model_with_tools = model.bind_tools([langchain_core_exists])


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# class GraphState(MessagesState):
#    pass


def explain_node(s: GraphState):
    message = model_with_tools.invoke(f"Explain the code: {s['messages'][-1].content}").content
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


def invoke_node(s: GraphState):
    s["messages"] = [model.invoke(s["messages"])]
    return s


builder = StateGraph(GraphState)

# builder.add_node("START", START)
# builder.add_node("END", END)

# builder.set_entry_point(START)
# builder.set_finish_point(END)


builder.add_node("tools", ToolNode([langchain_core_exists]))
builder.add_node(invoke_node.__name__, invoke_node)
builder.add_node(decide_request_kind.__name__, decide_request_kind)
builder.add_node(generate_node.__name__, generate_node)
builder.add_node(explain_node.__name__, explain_node)


builder.add_conditional_edges("explain_node", tools_condition)
builder.add_conditional_edges(START, decide_request_kind)

builder.add_edge("generate_node", END)

builder.add_edge("tools", "invoke_node")

builder.add_edge("invoke_node", END)

graph = builder.compile()

print(graph.get_graph().draw_ascii())

# Explain to me what builder.compile does in LangGraph
for chunk in graph.stream(
    {
        "messages": """explain the code :
Can ChatMistralAI be imported from langchain_core?
"""
    }
):
    print(json.dumps(chunk, indent=2))

# result = graph.invoke({"user_request": "Generate code to sort a list"})
# print(result)
