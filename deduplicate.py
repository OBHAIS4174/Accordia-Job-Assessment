import json

input_path = "all_formatted_combined.jsonl"
output_path = "all_clean_deduplicated.jsonl"

seen = set()
cleaned = []

def is_valid_message(messages):
    if not isinstance(messages, list) or len(messages) != 2:
        return False
    for m in messages:
        if "role" not in m or "content" not in m:
            return False
        content = m["content"]
        if content is None or content.strip().lower() == "nan" or content.strip() == "":
            return False
    return True

with open(input_path, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line)
        if "messages" not in data:
            continue
        if not is_valid_message(data["messages"]):
            continue
        key = json.dumps(data["messages"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            cleaned.append(data)

with open(output_path, "w", encoding="utf-8") as outfile:
    for item in cleaned:
        outfile.write(json.dumps(item) + "\n")

print(f"Cleaned & deduplicated: {len(cleaned)} valid unique records saved to {output_path}")
