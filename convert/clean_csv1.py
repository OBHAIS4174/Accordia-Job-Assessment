import pandas as pd
import json

# File paths
input_csv = "counsel_chat2.csv"  # Change this to match your CSV filename
output_jsonl = "formatted_counselchat.jsonl"

# Load CSV
df = pd.read_csv(input_csv)

# Open JSONL file for writing
with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for _, row in df.iterrows():
        question = str(row["questionText"]).strip().replace("\n", " ")
        answer = str(row["answerText"]).strip().replace("\n", " ")

        if question and answer:
            record = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ Converted {len(df)} rows to '{output_jsonl}'")
