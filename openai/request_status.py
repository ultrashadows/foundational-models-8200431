import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

batch = openai.batches.retrieve("batch_67aa1a0eee5081908cd21b0126b527fd")  # Replace with batch ID
print(batch)
