"""Submit an OpenAI Batch API job that writes exam-style questions for sampled ``files.stj`` decisions.

Prints the batch id. Pass it to ``request_status``, then pass the output file id to ``retrieve_batch``.
"""

import argparse
import json
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

from db import connect

HERE = Path(__file__).resolve().parent
MODEL = "gpt-4o-mini"


def load_prompt() -> str:
    """The prompt used for the thesis is kept in prompt.txt. OPENAI_PROMPT overrides it."""
    return os.getenv("OPENAI_PROMPT") or (HERE / "prompt.txt").read_text(encoding="utf-8").rstrip("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=50, help="documents per batch")
    parser.add_argument("--output", type=Path, default=HERE / "requests.jsonl", help="batch input file to write")
    args = parser.parse_args()

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = load_prompt()

    print("Fetching MongoDB documents...")
    documents = list(connect()["files"]["stj"].aggregate([
        {"$match": {
            "Texto Integral": {"$exists": True, "$ne": ""},
            "results": {"$exists": False},
            "wrong": {"$exists": False},
        }},
        {"$sample": {"size": args.sample_size}},
    ]))

    with open(args.output, "w", encoding="utf-8") as f:
        for doc in documents:
            request = {
                "custom_id": "request-" + str(doc["_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [{"role": "system", "content": prompt + doc["Texto Integral"]}],
                },
            }
            f.write(json.dumps(request, ensure_ascii=False).replace(" ", " ") + "\n")

    with open(args.output, "rb") as f:
        batch_input_file = openai.files.create(file=f, purpose="batch")

    batch_request = openai.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(batch_request.id)


if __name__ == "__main__":
    main()
