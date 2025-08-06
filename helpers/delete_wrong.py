from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["database"]
collection = db["test"]

result = collection.update_many(
    {"wrong": {"$exists": True}},
    {"$unset": {"wrong": ""}}
)

print(f"Removed 'wrong' field from {result.modified_count} documents.")
