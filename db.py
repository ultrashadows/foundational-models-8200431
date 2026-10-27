import os

from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")


def connect(**kwargs) -> MongoClient:
    """Return a client for the MongoDB instance that holds the crawled corpus."""
    return MongoClient(MONGO_URI, **kwargs)
