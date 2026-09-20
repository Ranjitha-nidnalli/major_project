"""
Invariant test (Task 9): every third-party import under backend/ resolves
to a package declared in backend/requirements.txt.

This guards against the exact class of defect fixed in Task 3: torch,
rank_bm25, rouge-score, psutil, openai, and indic-nlp-library were all
imported somewhere under backend/ but missing from requirements.txt.

Approach: statically parse every .py file under backend/ (excluding the
archived stale-eval directory) with ast, collect top-level import names,
subtract stdlib modules and local (in-repo) modules, and check every
remaining name maps to a package requirements.txt declares -- either
directly or, for a documented short list of packages, transitively via a
declared dependency (see TRANSITIVE_OK below). A package that is commented
out in requirements.txt (e.g. `# ollama>=0.1.7`) still counts as
"declared but optional" rather than missing, since llm_client.py only
imports `ollama` lazily behind an LLM_BACKEND=="ollama" check and the
comment documents why it's off by default.

This is a static check, not a substitute for actually installing
requirements.txt in a fresh venv (see Task 3's commit message for that
verification and its limits in this sandbox).
"""
import ast
import os
import re
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQUIREMENTS_PATH = os.path.join(BACKEND_DIR, "requirements.txt")

EXCLUDED_DIRS = {"_archive_stale", "__pycache__", ".git"}

# Import name -> distribution name(s) as they appear in requirements.txt.
# Only needed where the import name differs from the PyPI distribution name.
IMPORT_TO_DISTRIBUTION = {
    "dotenv": "python-dotenv",
    "indicnlp": "indic-nlp-library",
    "rouge_score": "rouge-score",
    "qdrant_client": "qdrant-client",
    "sentence_transformers": "sentence-transformers",
}

# Imports that are guaranteed to be installed transitively by a package that
# IS declared directly in requirements.txt, so their absence as a top-level
# requirements.txt line is not a defect. Documented explicitly rather than
# silently ignored.
TRANSITIVE_OK = {
    "pydantic": "installed transitively by fastapi>=0.110.0 (FastAPI requires "
                 "Pydantic v2); also declared directly as of this commit.",
}


def _local_module_names():
    """
    Every importable name that is defined inside backend/ itself -- not
    just top-level modules/packages, but also every .py filename stem
    anywhere under backend/, since several eval/ scripts import siblings
    directly (e.g. `from bm25_retriever import ...` from within
    backend/eval/), relying on the running script's own directory being on
    sys.path rather than a package-qualified import.
    """
    names = set()
    for entry in os.listdir(BACKEND_DIR):
        full = os.path.join(BACKEND_DIR, entry)
        if entry.endswith(".py") and os.path.isfile(full):
            names.add(entry[:-3])
        elif os.path.isdir(full) and entry not in EXCLUDED_DIRS:
            if os.path.isfile(os.path.join(full, "__init__.py")) or entry == "eval":
                # `eval/` is a namespace package (no __init__.py) but is
                # imported as `eval.numeric_faithfulness` elsewhere in the repo.
                names.add(entry)

    for root, dirs, files in os.walk(BACKEND_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for fn in files:
            if fn.endswith(".py"):
                names.add(fn[:-3])
    return names


def _collect_imports():
    top_level_imports = set()
    for root, dirs, files in os.walk(BACKEND_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read(), filename=path)
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top_level_imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:
                        top_level_imports.add(node.module.split(".")[0])
    return top_level_imports


def _declared_packages(requirements_text):
    """
    Returns (active_names, all_names_including_commented).
    Names are lowercased distribution names, extras like [standard] stripped.
    """
    active = set()
    all_declared = set()
    line_pattern = re.compile(r"^\s*#?\s*([A-Za-z0-9_.\-]+)")
    for raw_line in requirements_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_commented = line.startswith("#")
        content = line.lstrip("#").strip()
        if not content or content.startswith("-"):
            continue
        match = line_pattern.match(content)
        if not match:
            continue
        name = match.group(1)
        name = re.split(r"[\[<>=!~;]", name)[0].strip().lower()
        if not name:
            continue
        all_declared.add(name)
        if not is_commented:
            active.add(name)
    return active, all_declared


def test_every_third_party_import_resolves_from_requirements():
    stdlib_names = set(sys.stdlib_module_names) | {"__future__"}
    local_names = _local_module_names()

    imports = _collect_imports()
    third_party = imports - stdlib_names - local_names

    with open(REQUIREMENTS_PATH, encoding="utf-8") as f:
        requirements_text = f.read()
    active_declared, all_declared = _declared_packages(requirements_text)

    missing = []
    for import_name in sorted(third_party):
        if import_name in TRANSITIVE_OK:
            continue
        distribution = IMPORT_TO_DISTRIBUTION.get(import_name, import_name).lower()
        if distribution in all_declared:
            continue
        missing.append(import_name)

    assert not missing, (
        f"Imports with no corresponding requirements.txt entry (active or "
        f"commented-documented): {missing}. Either add them to "
        f"requirements.txt, or if they resolve transitively through another "
        f"declared package, add them to TRANSITIVE_OK in this test with a "
        f"one-line justification."
    )
