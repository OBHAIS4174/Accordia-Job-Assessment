import json 
import random 

with open('1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

output = []

for intent in data['intents']:
    tag = intent.get("tag")
    patterns = intent.get("patterns", [])
    responses = intent.get("resonses", [])

    for pattern in patterns:
        for response in responses:
            output.append({
                "messages": [
                    {"role": "user", "content": pattern.strip()},
                    {"role": "assistant", "content": response.strip()}
                ]
            })

with open('formated1.jsonl', 'w', encoding='utf-8') as f_out:
    for item in output:
        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ Converted {len(output)} training samples to 1_finetune.jsonl")