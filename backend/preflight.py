"""
Demo-day preflight check (TODO #9).

Confirms the two external services the backend depends on are actually up
and populated *before* a demo, instead of discovering a dead Qdrant
collection or an unreachable Mongo instance mid-demo. Deliberately does not
import vector_db.py or chat_db.py -- those pull in the BGE-M3 embedding
model / open long-lived connections as import side effects, which is slow
and unnecessary for a yes/no reachability check.

Usage:
    cd backend && python preflight.py

Exit code 0 if everything is reachable and non-empty, 1 otherwise -- usable
as a pre-demo gate in a shell script or CI job.
"""
import os
import sys

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "sugarcane_knowledge"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = "sugarcane_chat"


def check_qdrant():
    if not QDRANT_URL:
        print("FAIL  Qdrant: QDRANT_URL is not set -- would fall back to local "
              "file mode (ARCHITECTURE.md Section 6), not the demo's server mode.")
        return False

    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        collections = {c.name for c in client.get_collections().collections}
    except Exception as e:
        print(f"FAIL  Qdrant: could not connect to {QDRANT_URL} ({e})")
        return False

    if COLLECTION_NAME not in collections:
        print(f"FAIL  Qdrant: connected to {QDRANT_URL}, but collection "
              f"'{COLLECTION_NAME}' does not exist. Run `python vector_db.py` to seed it.")
        return False

    info = client.get_collection(COLLECTION_NAME)
    count = client.count(COLLECTION_NAME, exact=True).count
    if count == 0:
        print(f"FAIL  Qdrant: collection '{COLLECTION_NAME}' exists but has 0 points.")
        return False

    print(f"OK    Qdrant: {QDRANT_URL} reachable, collection '{COLLECTION_NAME}' "
          f"has {count} points, status={info.status}.")
    return True


def check_mongo():
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except PyMongoError as e:
        print(f"FAIL  MongoDB: could not reach {MONGO_URI} ({e})")
        return False

    db_names = client.list_database_names()
    print(f"OK    MongoDB: {MONGO_URI} reachable. "
          f"'{MONGO_DB_NAME}' {'exists' if MONGO_DB_NAME in db_names else 'does not exist yet (created on first chat message, this is fine)'}.")
    return True


def main():
    results = [check_qdrant(), check_mongo()]
    print()
    if all(results):
        print("Preflight passed -- ready to demo.")
        sys.exit(0)
    else:
        print("Preflight FAILED -- fix the above before demoing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
