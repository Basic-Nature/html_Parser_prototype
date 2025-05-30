import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Now BASE_DIR points to .../webapp
CONTEXT_DB_PATH = os.path.join(BASE_DIR, "parser", "Context_Integration", "Context_Library", "context_elections.db")
CONTEXT_LIBRARY_PATH = os.path.join(
    BASE_DIR, "parser", "Context_Integration", "Context_Library", "context_library.json"
)
