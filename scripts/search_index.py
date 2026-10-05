"""Ask the search index a question from the command line.

Run:  PYTHONPATH=src .venv/bin/python scripts/search_index.py "Wie wird Wasserstoff gefördert?"
"""

import sys

from energiewende.index import search, store

question = sys.argv[1]
index = store.load_index()
for hit in search.search(index, question, k=4):
    print(f"{hit['score']}  {hit['number']}  {hit['date']}  {hit['title'][:70]}")
    print("      " + hit["text"][:200].replace("\n", " "))
