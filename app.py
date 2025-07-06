import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

@st.cache_resource
def load_model():
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "./checkpoints/checkpoint-81000"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        offload_buffers=True,
    )
    
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    
    return model, tokenizer

model, tokenizer = load_model()

st.title("Mental Health Support Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""
if "current_state" not in st.session_state:
    st.session_state.current_state = "normal" 
if "last_bot_question" not in st.session_state:
    st.session_state.last_bot_question = ""

def detect_crisis_keywords(text):
    """Detect crisis-related keywords"""
    crisis_patterns = [
        r'\b(suicid|kill myself|end it all|don\'t want to live|hurt myself)\b',
        r'\b(self.harm|cutting|overdose|jump off|hang myself)\b',
        r'\b(no point|worthless|better off dead|can\'t go on)\b'
    ]
    
    text_lower = text.lower()
    for pattern in crisis_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def detect_positive_response(text):
    """Detect if user is responding positively to a question"""
    positive_patterns = [
        r'\b(yes|yeah|ok|okay|sure|tell me|please|help|want to talk)\b',
        r'\b(i want|i need|please help|go ahead)\b'
    ]
    
    text_lower = text.lower().strip()
    for pattern in positive_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def provide_crisis_response():
    """Provide immediate crisis support information"""
    st.session_state.current_state = "post_crisis"
    st.session_state.last_bot_question = "Would you like to talk about what's making you feel this way?"
    
    return """🚨 **I'm really concerned about you. Please reach out for immediate help:**

**Malaysia:**
- Befrienders KL: 03-7627-2929
- Talian Kasih: 15999
- Mental Health Psychosocial Support Service: 03-2935-9935

**International:**
- Crisis Text Line: Text HOME to 741741
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

**Remember:** You matter, and there are people who want to help you through this. 💙

Would you like to talk about what's making you feel this way, or would you prefer information about local mental health resources?"""

def create_contextual_prompt(user_msg, context="", state="normal"):
    """Create context-aware prompts based on conversation state"""
    
    if state == "post_crisis" and detect_positive_response(user_msg):
        system_prompt = """You are a compassionate mental health support chatbot. The user has just expressed suicidal thoughts and you provided crisis resources. They have now indicated they want to talk about what's making them feel this way.

Your response should:
- Be warm and supportive
- Ask gentle, open-ended questions about their feelings
- Show that you're listening and care
- Encourage them to share what's been difficult
- Use phrases like "I'm here to listen" and "Thank you for trusting me"

Do NOT:
- Give generic advice about self-harm
- Ask about methods or specific ways of self-harm
- Be clinical or detached
- Rush to give solutions immediately"""

        return f"""{system_prompt}

The user previously expressed wanting to hurt themselves, and when asked if they wanted to talk about what's making them feel this way, they responded: "{user_msg}"

Response: I'm so glad you're willing to talk with me. I'm here to listen without judgment. Can you tell me what's been going on that's making you feel this way? What's been weighing on your mind lately?"""

    elif state == "listening":
        system_prompt = """You are a compassionate mental health support chatbot. The user is sharing their feelings and struggles. 

Your response should:
- Validate their feelings
- Ask follow-up questions to understand better
- Show empathy and active listening
- Offer gentle support and coping strategies when appropriate
- Never minimize their pain"""

        return f"""{system_prompt}

Previous context: {context}

User's message: {user_msg}

Please respond with empathy and ask a gentle follow-up question to help them explore their feelings further."""

    else:
        system_prompt = """You are a compassionate mental health support chatbot. Provide empathetic, supportive responses while maintaining professional boundaries."""

        return f"""{system_prompt}

{context}

User: {user_msg}
Assistant:"""

def ask_model(prompt, max_new_tokens=200):
    """Generate response using the fine-tuned model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
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
    
    if "Response:" in decoded:
        response = decoded.split("Response:")[-1].strip()
    elif "Assistant:" in decoded:
        response = decoded.split("Assistant:")[-1].strip()
    else:
        response = decoded[len(prompt):].strip()
    
    response = re.sub(r'\n+', '\n', response)
    response = response.split('\n\n')[0]
    
    return response

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            if detect_crisis_keywords(prompt) and st.session_state.current_state != "post_crisis":
                response = provide_crisis_response()
                
            elif st.session_state.current_state == "post_crisis":
                if detect_positive_response(prompt):
                    st.session_state.current_state = "listening"
                    contextual_prompt = create_contextual_prompt(prompt, "", "post_crisis")
                    response = ask_model(contextual_prompt)
                    
                    if len(response.strip()) < 30:
                        response = "I'm so glad you're willing to talk with me. I'm here to listen without judgment. Can you tell me what's been going on that's making you feel this way? What's been weighing on your mind lately?"
                        
                elif "resources" in prompt.lower() or "information" in prompt.lower():
                    response = """Here are some mental health resources that might help:

**Professional Support:**
- Malaysian Mental Health Association: 03-2780-6803
- Hospital Kuala Lumpur Psychiatry Department: 03-2615-5555
- University Malaya Medical Centre: 03-7949-2068

**Online Resources:**
- MIASA (Mental Illness Awareness & Support Association): www.miasa.org.my
- Malaysian Psychiatric Association: www.psychiatry-malaysia.org

**Self-Care Tips:**
- Practice deep breathing exercises
- Try mindfulness or meditation
- Keep a journal of your thoughts and feelings
- Maintain regular sleep and eating schedules

Is there anything specific you'd like to know more about?"""
                    st.session_state.current_state = "normal"
                    
                else:
                    contextual_prompt = create_contextual_prompt(prompt, st.session_state.conversation_context, "normal")
                    response = ask_model(contextual_prompt)
                    st.session_state.current_state = "normal"
                    
            elif st.session_state.current_state == "listening":
                contextual_prompt = create_contextual_prompt(prompt, st.session_state.conversation_context, "listening")
                response = ask_model(contextual_prompt)
                
            else:
                contextual_prompt = create_contextual_prompt(prompt, st.session_state.conversation_context, "normal")
                response = ask_model(contextual_prompt)
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    new_context = f"User: {prompt}\nBot: {response[:100]}..."
    if len(st.session_state.conversation_context.split("User:")) > 6:
        parts = st.session_state.conversation_context.split("User:")
        st.session_state.conversation_context = "User:" + "User:".join(parts[-6:])
    st.session_state.conversation_context += f"\n{new_context}"

with st.sidebar:

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_context = ""
        st.session_state.current_state = "normal"
        st.session_state.last_bot_question = ""
        st.rerun()