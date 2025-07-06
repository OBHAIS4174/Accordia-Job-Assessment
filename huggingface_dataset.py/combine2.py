import json

# === Input files ===
file_1 = r"C:\Users\Omaar\OneDrive - Universiti Teknikal Malaysia Melaka\Desktop\ML\huggingface_dataset.py\mental_health_all_combined_deduplicated.jsonl"
file_2 = r"C:\Users\Omaar\OneDrive - Universiti Teknikal Malaysia Melaka\Desktop\ML\huggingface_dataset.py\all_clean_deduplicated.jsonl"

# === Load data from both files ===
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

data1 = load_jsonl(file_1)
data2 = load_jsonl(file_2)

# === Merge and deduplicate ===
combined = data1 + data2
unique_set = set()
deduplicated = []

for item in combined:
    try:
        user = item["messages"][0]["content"].strip().lower()
        assistant = item["messages"][1]["content"].strip().lower()
        key = (user, assistant)
        if key not in unique_set:
            unique_set.add(key)
            deduplicated.append(item)
    except Exception:
        continue  # skip malformed

# === Save the final combined file ===
output_path = "all_merged_deduplicated.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in deduplicated:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Merged and saved {len(deduplicated)} unique items to {output_path}")
