import json
import os
import openai
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

from embed.helpers import embed

load_dotenv()
client = MongoClient("mongodb://localhost:27017/")
db = client["files"]
collection = db["jtrl"]

openai.api_key = os.getenv('OPENAI_API_KEY')

response = openai.files.content('file-KbHQSF8gqUAZ41KSvwMN65').text.split('\n')  # Replace with file ID

for item in response:
    if item or item.strip() != '':
        data = json.loads(item)
        request_id = data['custom_id'].split('-')[1]
        result = data['response']['body']['choices'][0]['message']['content']
        if result.startswith('```json'):
            result = result[len('```json'):].strip()
        if result.endswith('```'):
            result = result[:-3].strip()

        try:
            parsed_result = json.loads(result)
            document = collection.find_one({"_id": ObjectId(request_id)})
            if document:
                most_distant = embed(document)
                collection.update_one(
                    {"_id": ObjectId(request_id)},
                    {"$set": {"wrong": most_distant["_id"], "results": parsed_result}}
                )
                print("Document updated successfully.")
            else:
                print("No document found with _id:", request_id)
        except json.JSONDecodeError:
            print(f"Skipping document due to JSON decoding error.")
            continue
        print(parsed_result)
