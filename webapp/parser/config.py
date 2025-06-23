import os

# PROJECT_ROOT is the parent directory of 'webapp', i.e., the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# BASE_DIR points to .../webapp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEXT_DB_PATH = os.path.join(BASE_DIR, "parser", "Context_Integration", "Context_Library", "context_elections.db")
CONTEXT_LIBRARY_PATH = os.path.join(
    BASE_DIR, "parser", "Context_Integration", "Context_Library", "context_library.json"
)
MODEL_DIR = os.path.dirname(BASE_DIR)
# Usage: for subprocesses, set cwd=PROJECT_ROOT and ensure PROJECT_ROOT is in PYTHONPATH

if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("BASE_DIR:", BASE_DIR)