"""Download the energy papers and build the search index.

Run from the project folder:  PYTHONPATH=src .venv/bin/python scripts/build_index.py [chunk_size]
"""

import sys
import time

from energiewende.index import store
from energiewende.ingest import dip

chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 400

papers = dip.get_energy_papers()
print(f"{len(papers)} papers downloaded")

start = time.time()
store.build_index(papers, chunk_size=chunk_size)
seconds = time.time() - start
print(f"{store.count_chunks(chunk_size)} chunks of size {chunk_size} saved in {seconds:.0f} seconds")
