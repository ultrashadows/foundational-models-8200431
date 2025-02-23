import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

response = openai.files.content('file-KbHQSF8gqUAZ41KSvwMN65').text.split('\n') # Replace with file ID

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
        except json.JSONDecodeError:
            print(f"Skipping document due to JSON decoding error.")
            continue
        print(parsed_result)
