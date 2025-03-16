import json
import os
from concurrent.futures import ThreadPoolExecutor

import openai
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

from embed.helpers import embed

load_dotenv()
client = MongoClient("mongodb://localhost:27017/")
db = client["files"]
collection = db["stj"]

openai.api_key = os.getenv('OPENAI_API_KEY')

response = openai.files.content('file-EHaf2kXmuqjQX1LpLaTSVQ').text.split('\n')  # Replace with file ID


def process_item(item, collection, documents, index, total):
    if item or item.strip() != '':
        data = json.loads(item)
        request_id = data['custom_id'].split('-')[1]
        print(f"Processing item {index + 1}/{total}")  # Count log

        result = data['response']['body']['choices'][0]['message']['content']

        if result.startswith('```json'):
            result = result[len('```json'):].strip()
        if result.endswith('```'):
            result = result[:-3].strip()

        try:
            parsed_result = json.loads(result)
            document = collection.find_one({"_id": ObjectId(request_id)})
            if document:
                most_distant = embed(document, documents)
                collection.update_one(
                    {"_id": ObjectId(request_id)},
                    {"$set": {"wrong": most_distant["_id"], "results": parsed_result}}
                )
                print(f"Document {request_id} updated successfully.")
            else:
                print(f"No document found with _id: {request_id}")
        except json.JSONDecodeError:
            print(f"Skipping document {request_id} due to JSON decoding error.")


def process_response(response, collection, max_threads=20):
    documents = list(collection.find())
    total_items = len(response)
    print(f"Total documents: {len(documents)}")
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(process_item, item, collection, documents, i, total_items) for i, item in
                   enumerate(response)]
        for future in futures:
            future.result()


process_response(response, collection)
