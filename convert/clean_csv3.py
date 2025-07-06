import pandas as pd
import json

# Load CSV (make sure to save the content into a file, e.g., 'context_response.csv')
df = pd.read_csv("Dataset.csv")

# Remove rows with missing values (optional but recommended)
df.dropna(subset=["Context", "Response"], inplace=True)

# Save to JSONL format
with open("formated_dataset.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        json.dump({
            "messages": [
                {"role": "user", "content": row["Context"].strip()},
                {"role": "assistant", "content": row["Response"].strip()}
            ]
        }, f, ensure_ascii=False)
        f.write("\n")

print("✅ Successfully saved as formated_dataset")
