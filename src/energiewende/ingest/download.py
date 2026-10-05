"""Download JSON from a URL and keep a copy in a cache folder."""

import hashlib
import json
import os
import time

import requests

from energiewende import config


def get_json(url, params=None, headers=None, max_age_hours=24):
    """Return the JSON at url. Use the cached copy if it is younger than max_age_hours."""
    # The file name is a hash of the URL and the parameters.
    # Headers are not part of it, so API keys never end up in file names.
    key = url + json.dumps(params or {}, sort_keys=True)
    file_name = hashlib.sha256(key.encode()).hexdigest() + ".json"
    path = os.path.join(config.CACHE_DIR, file_name)

    if os.path.exists(path):
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        if age_hours < max_age_hours:
            with open(path, encoding="utf-8") as f:
                return json.load(f)

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()  # stop with an error on 4xx / 5xx
    data = response.json()

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data
