"""Download a finished OpenAI batch and store the generated questions in ``files.stj``.

Each answered document gets ``results`` (the parsed list of ``pergunta``/``resposta`` items) and
``wrong`` (the ``_id`` of its most distant document, see ``helpers/embed.py``).
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import openai
from bson import ObjectId
from dotenv import load_dotenv

from db import connect
from helpers.embed import embed


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def process_item(item, collection, documents, index, total):
    if not item.strip():
        return
    data = json.loads(item)
    request_id = data["custom_id"].split("-")[1]
    print(f"Processing item {index + 1}/{total}")

    result = strip_code_fence(data["response"]["body"]["choices"][0]["message"]["content"])

    try:
        parsed_result = json.loads(result)
    except json.JSONDecodeError:
        print(f"Skipping document {request_id} due to JSON decoding error.")
        return

    document = collection.find_one({"_id": ObjectId(request_id)})
    if not document:
        print(f"No document found with _id: {request_id}")
        return

    most_distant = embed(document, documents, collection)
    collection.update_one(
        {"_id": ObjectId(request_id)},
        {"$set": {"wrong": most_distant["_id"], "results": parsed_result}},
    )
    print(f"Document {request_id} updated successfully.")


def process_response(response, collection, max_threads=20):
    documents = list(collection.find())
    total_items = len(response)
    print(f"Total documents: {len(documents)}")
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(process_item, item, collection, documents, i, total_items)
                   for i, item in enumerate(response)]
        for future in futures:
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file_id", help="output_file_id reported by request_status, for example file-...")
    args = parser.parse_args()

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.files.content(args.output_file_id).text.split("\n")
    process_response(response, connect()["files"]["stj"])


if __name__ == "__main__":
    main()
