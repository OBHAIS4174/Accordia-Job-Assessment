import json

# File paths
input_file = 'intents.json'         # Your original file (replace with correct filename)
output_file = 'formatted_intents.jsonl'  # Output for fine-tuning

# Load data
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process each intent pattern-response pair into chat format
with open(output_file, 'w', encoding='utf-8') as fout:
    for intent in data["intents"]:
        tag = intent["tag"]
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])

        for pattern in patterns:
            for response in responses:
                chat_sample = {
                    "messages": [
                        {"role": "user", "content": pattern.strip()},
                        {"role": "assistant", "content": response.strip()}
                    ]
                }
                json.dump(chat_sample, fout, ensure_ascii=False)
                fout.write('\n')

print(f"✅ Converted to {output_file}")
