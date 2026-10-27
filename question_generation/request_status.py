"""Print the status of an OpenAI batch job (its ``output_file_id`` appears once it completes)."""

import argparse
import os

import openai
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id", help="id printed by create_request, for example batch_...")
    args = parser.parse_args()

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print(openai.batches.retrieve(args.batch_id))


if __name__ == "__main__":
    main()
