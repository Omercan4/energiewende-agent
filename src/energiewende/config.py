"""All settings in one place. Values come from environment variables or the .env file."""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the current folder, if it exists

# Folder where downloaded API responses are cached.
CACHE_DIR = os.environ.get("CACHE_DIR", "data/cache")

# Key for the Bundestag DIP API.
DIP_API_KEY = os.environ.get("DIP_API_KEY", "")
