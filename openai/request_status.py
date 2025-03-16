import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

batch = openai.batches.retrieve("batch_67d41e2e921c8190b712e74828632dc2")  # Replace with batch ID
print(batch)
