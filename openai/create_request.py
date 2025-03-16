import json
import os

import openai
from dotenv import load_dotenv
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["files"]
collection = db["stj"]

print("Fetching MongoDB documents...")
documents = list(collection.aggregate([
    {"$match": {
        "Texto Integral": {"$exists": True, "$ne": ""},
        "results": {"$exists": False},
        "wrong": {"$exists": False}
    }},
    {"$sample": {"size": 50}}
]))

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
prompt = os.getenv('OPENAI_PROMPT')

batch_requests = []

for doc in documents:
    doc_decision = doc["Texto Integral"]
    if doc_decision:
        batch_requests.append({
            "custom_id": "request-" + str(doc["_id"]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": prompt + doc_decision
                    }
                ]
            }
        })

with open("requests.jsonl", "w", encoding="utf-8") as f:
    for item in batch_requests:
        json_str = json.dumps(item, ensure_ascii=False)
        json_str = json_str.replace("\u00A0", " ")
        f.write(json_str + "\n")

batch_input_file = openai.files.create(
    file=open("requests.jsonl", "rb"),
    purpose="batch"
)

file_id = batch_input_file.id

batch_request = openai.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

print(batch_request.id)
