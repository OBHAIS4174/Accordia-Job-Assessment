import pandas as pd
import json
from bs4 import BeautifulSoup


input_csv = "counselchat-data.csv"  
output_jsonl = "formatted_mental_health_2.jsonl"

df = pd.read_csv(input_csv)

def clean_html(text):
    return BeautifulSoup(str(text), "html.parser").get_text().strip()

with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for _, row in df.iterrows():
        question = clean_html(row.get("questionText", ""))
        answer = clean_html(row.get("answerText", ""))

        if question and answer:
            json.dump({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }, f_out, ensure_ascii=False)
            f_out.write("\n")

print(f"✅ Saved fine-tuning dataset as '{output_jsonl}'")
