import os
import sys

# Make `backend/` importable as top-level modules (rag_service, vector_db, etc.)
# the same way the eval/ scripts do it, so tests can `import` project modules
# without needing an installed package.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
