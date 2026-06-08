import json

with open('test.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

print(len(eval_data))