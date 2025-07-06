from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch
from transformers import TrainerCallback

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    inference_mode=False,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

for name, param in model.named_parameters():
    if 'lora_' in name:
        param.requires_grad = True
        print(f"✅ Enabled gradients for: {name}")

dataset = load_dataset("json", data_files="all_merged_deduplicated.jsonl")["train"]
dataset = dataset.shuffle(seed=42).select(range(100000))
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# === RTX 2050 Optimized TrainingArguments ===
training_args = TrainingArguments(
    per_device_train_batch_size=2,        
    gradient_accumulation_steps=4,      
    learning_rate=5e-5,
    logging_steps=100,
    output_dir="./checkpoints",
    save_steps=1000,
    eval_strategy="no",               
    eval_steps=1000,
    save_total_limit=1,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,                      
    report_to="none",
    dataloader_pin_memory=False,          
    remove_unused_columns=True,
    optim="adamw_torch",
    dataloader_num_workers=0,            
    max_grad_norm=1.0,
    gradient_checkpointing=False,        
    warmup_steps=100,
    max_steps=100000,                      
)

class PrintEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nEpoch {int(state.epoch)} finished.\n")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 500 == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                print(f"Step {state.global_step}: GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# === SFTTrainer ===
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    peft_config=peft_config,
    callbacks=[PrintEpochCallback()],
)

if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    cached_memory = torch.cuda.memory_reserved(0) / 1024**3
    free_memory = total_memory - cached_memory
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {total_memory:.2f}GB")
    print(f"Allocated: {allocated_memory:.2f}GB")
    print(f"Cached: {cached_memory:.2f}GB")
    print(f"Free: {free_memory:.2f}GB")
    
    if free_memory < 1.0:
        print("Warning: Less than 1GB free VRAM. Consider reducing batch size.")

print("💡 RTX 2050 Settings:")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   - Max steps: {training_args.max_steps}")
print(f"   - Gradient checkpointing: {training_args.gradient_checkpointing}")

try:
    trainer.train()
    print("\nTraining completed successfully!")
    
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Model saved to")
    
except torch.cuda.OutOfMemoryError as e:
    print(f"\nGPU Out of Memory Error: {e}")

    
except RuntimeError as e:
    if "grad" in str(e):
        print(f"\nGradient Error: {e}")
        print("💡 This usually means gradient issues. Check LoRA configuration.")
    else:
        print(f"\nRuntime Error: {e}")
        
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    try:
        trainer.save_model("./interrupted_model")
        tokenizer.save_pretrained("./interrupted_model")
        print("Model saved to ./interrupted_model")
    except:
        print("Could not save model")
        
except Exception as e:
    print(f"\nUnexpected error: {e}")
    print("Attempting to save current progress...")
    try:
        trainer.save_model("./error_model")
        tokenizer.save_pretrained("./error_model")
        print("Model saved to ./error_model")
    except:
        print("Could not save model")

finally:
    torch.cuda.empty_cache()
    print("\nTraining script completed!")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"Final GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")