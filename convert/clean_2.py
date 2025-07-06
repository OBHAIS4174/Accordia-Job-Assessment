import json

input_file = 'combined_dataset.json'        # Your source file
output_file = 'formatted_dataset.jsonl'     # Output for fine-tuning

# Load input data
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

# Convert each entry into chat format
with open(output_file, 'w', encoding='utf-8') as f_out:
    for entry in data:
        user_msg = entry.get("Context", "").strip()
        assistant_msg = entry.get("Response", "").strip()
        if user_msg and assistant_msg:
            formatted = {
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            }
            json.dump(formatted, f_out, ensure_ascii=False)
            f_out.write('\n')

print(f"✅ Done! Saved to {output_file}")
