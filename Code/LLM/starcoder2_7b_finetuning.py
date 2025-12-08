import os
import torch
import pandas as pd
import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from accelerate import Accelerator

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.cuda.empty_cache()
gc.collect()

def print_gpu_usage(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] Allocated: {a:.2f} GB | Reserved: {r:.2f} GB")

class GPUUsageCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print_gpu_usage(f"Step {state.global_step}")

class LossRecorderCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self.eval_losses.append(metrics["eval_loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_losses.append(logs["loss"])

# ======== Dataset ========
train_path = "dataset/train.xlsx"
val_path = "dataset/val.xlsx"

train_df = pd.read_excel(train_path)
val_df = pd.read_excel(val_path)

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(val_df)

model_path = "bigcode/starcoder2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    prompts = ["Give me a Flexscript. " + p for p in examples["Prompt"]]
    responses = examples["Response"]
    full_texts = [p + r for p, r in zip(prompts, responses)]
    tok = tokenizer(full_texts, padding="max_length", truncation=True, max_length=4096)
    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    labels = []
    for ids, p_ids in zip(tok["input_ids"], prompt_ids):
        plen = len(p_ids)
        labels.append([-100] * plen + ids[plen:])
    tok["labels"] = labels
    return tok

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["Prompt", "Response"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["Prompt", "Response"])


# ======== Model ========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map={"": accelerator.process_index},
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# ======== Training ========
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    logging_steps=10,
    fp16=True,
    report_to="none",
    load_best_model_at_end=True,
    gradient_checkpointing=False,
    learning_rate=1e-4,
    lr_scheduler_type="constant",
    warmup_steps=10,
    ddp_find_unused_parameters=False,
)

loss_callback = LossRecorderCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[GPUUsageCallback(), loss_callback],
    tokenizer=tokenizer,
)

trainer.train()

# ======== Save model ========
if torch.distributed.get_rank() == 0 or not torch.distributed.is_initialized():
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")
