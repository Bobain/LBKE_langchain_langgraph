"""## Routeur explication/génération de code avec LangGraph

On va créer un agent qui décide s'il doit générer du code, ou expliquer un code existant.
"""

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
# https://mistral.ai/products/la-plateforme#pricing section "cloud"
model = init_chat_model(
    "codestral-latest",
    model_provider="mistralai",
)


from typing import TypedDict


class RouterState(TypedDict):
    user_request: str
    response: str


def explain_node(s: RouterState) -> RouterState:
    return {"response": model.invoke(f"Explain the code: {s['user_request']}").content}


def generate_node(s):
    return {"response": model.invoke(f"Generate code for: {s['user_request']}").content}


from typing import Literal


def decide_request_kind(s: RouterState) -> Literal["generate_node", "explain_node"]:
    kind = model.invoke(
        f"Is this query a request to generate code, or explain code? Respond with 'explain' or 'generate'. Query: {s['user_request']}"
    ).content
    if kind == "generate":
        return "generate_node"
    else:
        return "explain_node"


# Bonus: tagging
# On a une garantie plus forte de la structure de la réponse
# (sinon l'appel plante)
from typing import Literal


class GenerateOrExplain(TypedDict):
    kind: Literal["generate", "explain"]


from dataclasses import dataclass


@dataclass
class GenerateOrExplain:
    kind: Literal["generate", "explain"]


# 3rd alternative : pydantic


def decide_request_kind_v3(s: RouterState) -> Literal["generate_node", "explain_node"]:
    model_structured = model.with_structured_output(GenerateOrExplain)
    res = model_structured.invoke(
        f"Is this query a request to generate code, or explain code? Respond with 'explain' or 'generate'. Query: {s['user_request']}"
    )
    if res["kind"] == "generate":
        return "generate_node"
    else:
        return "explain_node"


print(decide_request_kind_v3(RouterState(user_request="Does 'retrievers' exist in langchain_core module?")))
print(decide_request_kind_v3(RouterState(user_request="dakaldklklkak kal kdlakd lakd alk")))

from langgraph.graph import START, END, StateGraph

builder = StateGraph(RouterState)
builder.add_node("explain_node", explain_node)
builder.add_node("generate_node", generate_node)
builder.add_conditional_edges(START, decide_request_kind)
builder.add_edge("explain_node", END)
builder.add_edge("generate_node", END)
graph_basique = builder.compile()

# Was leading to a 443 somehow
# display(Image(graph.get_graph().draw_mermaid_png()))*
# Doesn't work
# %pip install -qU grandalf
# display(Image(graph.get_graph().draw_ascii()))

from IPython.display import Markdown

res = graph_basique.invoke({"user_request": "Explain to me what builder.compile does in LangGraph"})
Markdown(res["response"][0:500])

# Bonus : streaming pour voir les appels
import json

for chunk in graph_basique.stream({"user_request": "Explain to me what builder.compile does in LangGraph"}):
    print(json.dumps(chunk, indent=2))

"""## Agent complet avec messages et tool calling

### Etape 1 : gérer les messages multiples avec MessagesState
"""

from langgraph.graph import MessagesState


# We extend "MessagesState"
# which defines and handles "messages" field
class RouterState(MessagesState):
    pass


# We use pass because there is no other field
# To add more fields, remove "pass" and add them here


def explain_node_v2(s: RouterState) -> RouterState:
    # On demande au modèle d'expliquer le code fourni dans le message de l'utilisateur
    latest_user_message = s["messages"][-1].content
    return {"messages": [model.invoke(f"Explain the code: {latest_user_message}")]}


# Debug
print(explain_node_v2(RouterState(messages=[HumanMessage("Does 'retrievers' exist in langchain_core module?")])))


#
def generate_node_v2(s: RouterState):
    # On passe la liste des messages à ajouter (pas de tous les messages)
    # (on n'utilise pas d'outil ici pour l'instant)
    return {"messages": [model.invoke(f"Generate code for: {s['messages'][-1].content}")]}


from typing import Literal


def decide_request_kind_v2(s: RouterState) -> Literal["generate_node", "explain_node"]:
    kind = model.invoke(
        f"Is this query a request to generate code, or explain code? Respond with 'explain' or 'generate'. Query: {s['messages'][-1].content}"
    ).content
    print(f"Request kind decision {kind}")
    if kind == "generate":
        return "generate_node"
    else:
        return "explain_node"


# Bonus : arête de décision avec un prompt plus élaboré
from langchain_core.prompts import PromptTemplate

# If we remove "without quotes", it will sometime output 'explain' and not explain
# also we can change the words of the prompt and observe the retry mechanism
decision_prompt = PromptTemplate.from_template(
    """
Is this query a request to generate code, or explain code? Respond with 'explain' or 'generate', without quotes. Query:{query}
"""
)


def check_answer(answer):
    if answer.content in ["generate", "explain"]:
        return answer
    print(
        f"Warning: answer is not generate or explain: {answer}. Model may add unexpected punctuation or fail to answer."
    )
    raise Exception("Incorrect decision")


router_chain = (decision_prompt | model | check_answer).with_retry()
router_chain.invoke({"query": "Does 'retrievers' exist in langchain_core module?"})

# Autre approche possible : utiliser "with_structured_output" offrirait un garantie encore plus forte d'obtenir une réponse "explain" ou "generate"

# Same graph as earlier, should work the same but with standard "messages" field
# instead of "response" used earlier

# On ne configure pas encore l'appel d'outils

builder = StateGraph(RouterState)
builder.add_node("explain_node", explain_node_v2)
builder.add_node("generate_node", generate_node_v2)
builder.add_conditional_edges(START, decide_request_kind_v2)
builder.add_edge("explain_node", END)
builder.add_edge("generate_node", END)
graph_messages = builder.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))

from IPython.display import Markdown

res = graph_messages.invoke({"messages": [HumanMessage("Explain to me what builder.compile does in LangGraph")]})
# Only this line changed to get the right field
res

"""### Etape 2 : utiliser les noeuds clé-en-main de LangGraph pour gérer les outils"""

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


# New generic node that calls the LLM with all messages
def invoke_node(s: RouterState):
    s["messages"] = [model.invoke(s["messages"])]
    return s


def explain_node_v3(s: RouterState) -> RouterState:
    # version avec des outils
    pass
    # On demande au modèle d'expliquer le code fournit dans le message de l'utilisateur
    latest_user_message = s["messages"][-1].content
    return {"messages": [model_with_tools.invoke(f"Explain the code: {latest_user_message}")]}


# Debug
print(explain_node_v3(RouterState(messages=[HumanMessage("Does 'retrievers' exist in langchain_core module?")])))

# Now with proper tool calling

builder = StateGraph(RouterState)
# Decide explain or generate
builder.add_node("explain_node", explain_node_v3)  # Nouvelle version avec outils
builder.add_node("generate_node", generate_node_v2)
builder.add_node("invoke_node", invoke_node)
builder.add_conditional_edges(START, decide_request_kind_v2)
# If generate: done
builder.add_edge("generate_node", END)
# If explain: add tool nodes

# Redirige vers le noeud "tools" si un appel d'outil est détecté
builder.add_conditional_edges("explain_node", tools_condition)
# Exécute l'outil et génère le ToolMessage approprié
builder.add_node("tools", ToolNode([langchain_core_exists]))
# Noeud générique qui rappelle le LLM, avec les résultats fournis par l'outil
builder.add_edge("tools", "invoke_node")
builder.add_edge("invoke_node", END)

graph_tool = builder.compile()
# display(Image(graph_tool.get_graph().draw_mermaid_png()))

from IPython.display import Markdown

# res=graph.invoke({"user_request": "Explain to me what is LCEL in LangChain, in one sentence."})
res = graph_tool.invoke({"messages": "Does 'retrievers' exist in langchain_core module?"})

for m in res["messages"]:
    # print(m)
    m.pretty_print()

# Bonus : exemple avec une configuration, on pourrait passer un autre modèle
# On pourrait créer un modèle qui intègre notre RAG de la documentation LangChain
# res=graph_tool.invoke({
#     "messages": "Does 'retrievers' exist in langchain_core module?"},
#  { "configurable": {"model": model_with_tools}})

"""## Vers la v1 : agent ReAct avec create_agent"""

# Attention ce code ne fonctionnera qu'avec la v1 (encore en alpha au 09/2025)
from langchain.agents import create_agent

agent = create_agent(
    "mistra:codestral-latest",
    tools=[langchain_core_exists],
    prompt="You are a coding assistant, specialized in LangChain",
)
agent.invoke({"messages": [("user", "Does 'retrievers' exist in LangChain core?")]})

"""## Mémoire court terme"""

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
builder_mem = StateGraph(RouterState)
builder_mem.add_node(invoke_node)
builder_mem.add_edge(START, invoke_node.__name__)
graph_memory = builder_mem.compile(checkpointer=checkpointer)

import uuid

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}
res = graph_memory.invoke({"messages": ["What's the difference between LangSmith and LangFuse?"]}, config)
print(res["messages"][-1].content)

res2 = graph_memory.invoke({"messages": ["What was my last question?"]}, config)
print(res2["messages"][-1].content)

"""## Bonus : créer run outil à partir d'une chaîne LangChain RAG

"""

# Bonus : créer un outil à partir du RAG précédent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai import MistralAIEmbeddings

# /!\ Mistral embedding models doesn't support dimensions with "coarse-to-fine" approach
# as open AI does!
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    # should match your API limits
    max_concurrent_requests=6,
)
vector_store = InMemoryVectorStore(embeddings)
prompt = PromptTemplate.from_template(
    """
    Answer the query based on the retrieved documents.
    <query>{query}</query>
    <documents>{documents}</query>
    """
)
vector_store.add_texts(
    [
        "LangGraph is a framework to create agents.",
        "LangSmith is an LLM observability platform.",
        "init_chat_model allows to initialize chat models.",
    ]
)


def search_doc(query: str):
    closest_documents = vector_store.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(query)
    # print(closest_documents)
    # Fusion des chaînes de caractère
    documents = "\n\n".join([doc.page_content for doc in closest_documents])
    return {"query": query, "documents": documents}


rag_chain = search_doc | prompt | model
rag_chain.invoke("What is LangGraph ?").content

rag_tool = rag_chain.as_tool(
    name="LangChain-documentation-RAG", description="A function that can search for answers in LangChain documentation."
)
model_with_rag = model.bind_tools([langchain_core_exists, rag_tool])

model_with_rag.invoke("What is LangChain, according to its documentation?")

res = graph_tool.invoke(
    {
        "messages": """
    What does 'model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")' do in LangChain? Answer based on documents.
    """
    },
    {"configurable": {"model": model_with_rag}},
)
for msg in res["messages"]:
    msg.pretty_print()
