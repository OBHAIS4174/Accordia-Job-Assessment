# 🧠 Mental Health Support Chatbot

**Job Assessment Submission – Accordia Global Sdn Bhd**  
Author: *Omar A. A. Bhais*

---

## 📌 Overview

A compassionate AI-powered chatbot built using TinyLlama-1.1B-Chat, designed to provide mental health support, crisis intervention, and emotional wellness guidance. The model is fine-tuned on real mental health conversations using PEFT with LoRA to ensure low-resource adaptability and ethical support delivery.

---

## 🏥 Use Case

**Industry:** Healthcare  
**Problem:** Lack of affordable, 24/7 mental health support  
**Solution:** AI chatbot capable of:
- Recognizing emotional distress
- Offering coping strategies
- Escalating crisis scenarios with emergency contacts

---

## 🧠 Model Architecture

- **Base Model**: `TinyLlama-1.1B-Chat-v1.0`
- **Fine-Tuning**: PEFT (LoRA)
- **Dataset Sample**: 100k samples (from ~800k total)
- **Final Accuracy**: `~70.5%` mean token accuracy  
- **Loss**: `~1.085`

### Key Training Parameters:
- `batch_size=2`, `gradient_accumulation=4`
- `learning_rate=5e-5`, `fp16=True`
- `max_steps=100000`, trained up to step 81000 (~3.6 epochs)

---

## 🗂️ Project Structure

chatbot_assessment/
├── app.py                        # Streamlit frontend demo
├── test.py                       # Terminal-based interaction test
├── train.py                      # Fine-tuning script (TinyLlama + LoRA)
├── deduplicate.py                # Data deduplication utility
├── all_merged_deduplicated.jsonl# Final dataset used for training
├── checkpoints/                  # Fine-tuned model checkpoints (PEFT/LoRA)
├── dataset_cleaned/             # Cleaned dataset variants
├── dataset_uncleaned/           # Raw or source datasets
├── huggingface_dataset.py/      # Optional HuggingFace dataset loader
├── requirements.txt             # Dependencies for running locally
└── README.md                    # Project documentation


## 🚀 How to Run

### 1. Setup Environment
Create Virtual Enivronment 
```bash
pip install -r requirements.txt
2. Run the Frontend (Streamlit)
streamlit run app.py
3. Run a CLI Test
python test.py

🧩 Capabilities
✅ Crisis keyword detection & response

✅ Emotionally aware support (anxiety, stress, sadness)

✅ Personalized mental health strategies

✅ Malaysia-specific emergency numbers (Befrienders KL, Talian Kasih)

🧪 Example Interaction
Input:


I feel like I want to kill myself.
Output:

🚨 I'm really concerned about you. Please reach out to:
- Befrienders KL: 03-7627-2929
- Talian Kasih: 15999
Would you like to talk more about what’s making you feel this way?

Trained on limited hardware (RTX 2050, 4GB VRAM)

Developed by: Omar A. A. Bhais
For: Accordia Global Sdn Bhd - AI/ML Job Assessment
