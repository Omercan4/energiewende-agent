"""The agent: an LLM that can call tools in a loop until it can answer.

    question -> model -> (wants tools?) -> tools -> model -> ... -> answer

The model decides which tools to call. The tools run in Python and their
results go back to the model. This repeats until the model answers without
asking for a tool, or until MAX_STEPS is reached.
"""

import functools
import json
import time
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from energiewende import config
from energiewende.agent import tools

MAX_STEPS = 10  # graph steps per question (each model call and each tool run is one step)
PROMPT_VERSION = 2  # logged with each evaluation run; raise it when SYSTEM_PROMPT changes

SYSTEM_PROMPT = """You answer questions about the German energy system.
Today is {today}.

Use the tools:
- price_series, generation_load and weather for numbers (prices, power generation, consumption, weather).
- bundestag_search for what the Bundestag, the government or the parties plan, want or criticize.

Rules:
- Every number in your answer must come from a tool result. Use the mean, min, max and total
  that the tools give you. Never calculate or guess numbers yourself.
- If the question has several parts, call the tools for every part before you answer.
  Never answer that you still need another query.
- When you use Bundestag papers, name the Drucksache number.
- If a tool returns an error, say briefly what went wrong.
- If the question is not about energy in Germany, say that you can only answer energy questions.
- Answer in the language of the question. Keep it short."""


def get_llm():
    """The chat model behind any OpenAI-compatible URL."""
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
        temperature=0,
        timeout=60,
    )


def build_agent(llm, tool_list):
    """The loop: model -> tools -> model -> ... until the model answers."""
    model = llm.bind_tools(tool_list)  # tell the model which tools exist

    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)  # the state is the list of messages so far
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tool_list, handle_tool_errors=True))  # errors go back to the model
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)  # tool calls? -> "tools", else -> end
    builder.add_edge("tools", "model")
    return builder.compile()


@functools.cache
def default_agent():
    """The real agent with all tools, built once."""
    return build_agent(get_llm(), tools.TOOLS)


def ask(question, agent=None, today=None):
    """Run the agent on one question. Returns the answer and what happened on the way."""
    if agent is None:
        agent = default_agent()
    today = today or date.today()
    messages = [SystemMessage(SYSTEM_PROMPT.format(today=today)), HumanMessage(question)]

    start = time.time()
    try:
        messages = agent.invoke({"messages": messages}, {"recursion_limit": MAX_STEPS})["messages"]
        answer = messages[-1].text
    except GraphRecursionError:
        answer = "Sorry, this question needed too many steps. Please ask something more specific."
    latency_ms = int((time.time() - start) * 1000)

    tool_calls, sources, tokens = [], [], 0
    for message in messages:
        for call in getattr(message, "tool_calls", []):
            tool_calls.append({"name": call["name"], "args": call["args"]})
        if getattr(message, "usage_metadata", None):
            tokens += message.usage_metadata["total_tokens"]
        if message.type == "tool" and message.status == "success":
            for source in json.loads(message.content)["sources"]:
                if source not in sources:
                    sources.append(source)

    return {"answer": answer, "sources": sources, "tool_calls": tool_calls, "latency_ms": latency_ms, "tokens": tokens}
