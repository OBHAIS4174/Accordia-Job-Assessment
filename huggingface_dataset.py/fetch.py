from datasets import load_dataset
import json

# === Load datasets ===
ds_a = load_dataset("ZahrizhalAli/mental_health_conversational_dataset")["train"]
ds_b = load_dataset("mpingale/mental-health-chat-dataset")["train"]
ds_c = load_dataset("ShivomH/Mental-Health-Conversations")["train"]

conversations = []

# === Process Dataset A ===
for row in ds_a:
    text = row.get("text", "")
    if isinstance(text, str) and "<HUMAN>:" in text and "<ASSISTANT>:" in text:
        try:
            user = text.split("<HUMAN>:")[1].split("<ASSISTANT>:")[0].strip()
            assistant = text.split("<ASSISTANT>:")[1].strip()
            if user and assistant:
                conversations.append({
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant}
                    ]
                })
        except Exception:
            continue

# === Process Dataset B ===
for row in ds_b:
    question = row.get("questionText")
    answer = row.get("answerText")
    if isinstance(question, str) and isinstance(answer, str):
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            conversations.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })

# === Process Dataset C (ShivomH) ===
for row in ds_c:
    question = row.get("input")
    answer = row.get("response")
    if isinstance(question, str) and isinstance(answer, str):
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            conversations.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })

# === Deduplicate by lowercase content of user + assistant
unique_set = set()
deduplicated = []
for conv in conversations:
    try:
        msg_pair = (
            conv["messages"][0]["content"].strip().lower(),
            conv["messages"][1]["content"].strip().lower()
        )
        if msg_pair not in unique_set:
            unique_set.add(msg_pair)
            deduplicated.append(conv)
    except:
        continue

# === Save to JSONL ===
output_path = "mental_health_all_combined_deduplicated.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in deduplicated:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Total unique conversations saved: {len(deduplicated)} → {output_path}")
