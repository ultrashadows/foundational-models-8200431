import os

import openai
from pymongo import MongoClient
import json
from embed.helpers import embed
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
prompt = os.getenv('OPENAI_PROMPT')

client = MongoClient("mongodb://localhost:27017/")
db = client["files"]
collection = db["jtrl"]

print("Fetching MongoDB documents...")
documents = list(collection.aggregate([
    {"$match": {"Decisão Texto Integral": {"$exists": True}}},
    {"$sample": {"size": 1}}
]))

results = []
batch_messages = []
doc_references = []

for doc in documents:
    doc_decision = doc.get("Decisão Texto Integral", "").strip()
    if doc_decision:
        batch_messages.append({"role": "user", "content": prompt + doc_decision})
        doc_references.append(doc)

print(f"Processing {len(batch_messages)} documents...")

try:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=batch_messages
    )

    print(response)

    for i, choice in enumerate(response.choices):
        print(f"Handling document {i}")
        json_response = choice.message.content.strip()

        if json_response.startswith('```json'):
            json_response = json_response[len('```json'):].strip()
        if json_response.endswith('```'):
            json_response = json_response[:-3].strip()

        try:
            parsed_json = json.loads(json_response)
            most_distant = embed(doc_references[i])
            results.append({
                "results": parsed_json,
                "correct": str(doc_references[i]["_id"]),
                "wrong": str(most_distant["_id"])
            })
        except json.JSONDecodeError:
            print(f"Skipping document {i} due to JSON decoding error.")
            continue
except Exception as e:
    print(f"OpenAI request failed: {e}")

with open("../result.json", "w", encoding="utf-8") as result_file:
    json.dump(results, result_file, ensure_ascii=False, indent=4)

print("Results saved in result.json.")
