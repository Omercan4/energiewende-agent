"""Ask the agent a question from the command line.

Run:  PYTHONPATH=src .venv/bin/python scripts/ask.py "Wie hoch war der Strompreis gestern?"
"""

import sys

from energiewende.agent import graph

result = graph.ask(sys.argv[1])
print(result["answer"])
print()
print("Tools:  ", [call["name"] for call in result["tool_calls"]])
print("Sources:")
for source in result["sources"]:
    print("  -", source)
print(f"Time: {result['latency_ms']} ms   Tokens: {result['tokens']}")
