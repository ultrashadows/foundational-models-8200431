from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["database"]
collection = db["pool"]

result = collection.update_many(
    {"embedding": {"$exists": True}},
    {"$unset": {"embedding": ""}}
)

print(f"Removed 'wrong' field from {result.modified_count} documents.")
