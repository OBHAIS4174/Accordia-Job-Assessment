from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./checkpoints/checkpoint-81000"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base, adapter_path)
model.eval()
print("Model loaded successfully!\n")

def ask_model(prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's reply
    if "### Assistant:" in decoded:
        reply = decoded.split("### Assistant:")[-1]
        reply = reply.split("### User:")[0].strip()
        return reply
    else:
        return "[No assistant response found]"

def create_improved_prompt(user_message):
    return f"""### Instruction: You are a compassionate mental health support chatbot. Follow these guidelines:

1. CRISIS SITUATIONS: If someone mentions suicide, self-harm, or not wanting to live, immediately provide crisis resources:
   - "I'm very concerned about you. Please reach out to a crisis helpline immediately."
   - "In the US: 988 Suicide & Crisis Lifeline"
   - "In emergencies, call 911 or go to your nearest emergency room"

2. MEDICAL LIMITATIONS: You cannot diagnose, prescribe medication, or provide medical advice. Always recommend consulting healthcare professionals for medical concerns.

3. SCOPE: Stay focused on mental health support. For off-topic questions, gently redirect: "I'm here to help with mental health topics. What's on your mind today?"


4. SUPPORT APPROACH: Be empathetic, ask clarifying questions, and provide practical coping strategies when appropriate.

### User: {user_message}
### Assistant:"""


print("Question: It's just constant arguing and tension. My parents fight all the time, and it feels like I'm always caught in the middle. I feel responsible for fixing everything, but it's exhausting. I just want it to stop.")
offtopic_prompt = create_improved_prompt("It's just constant arguing and tension. My parents fight all the time, and it feels like I'm always caught in the middle. I feel responsible for fixing everything, but it's exhausting. I just want it to stop.")
offtopic_response = ask_model(offtopic_prompt)
print(f"🤖 Response: {offtopic_response}")
