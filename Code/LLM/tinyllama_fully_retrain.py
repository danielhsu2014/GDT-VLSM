import os
import torch
import pandas as pd
import gc
from datasets import Dataset
from transformers import (
    LlamaTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback
)
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.cuda.empty_cache()
gc.collect()
os.system("nvidia-smi")
print("TinyLLaMA Retrain Container started.")

def print_gpu_usage(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] GPU Memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

class GPUUsageCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print_gpu_usage(f"Step {state.global_step}")

class TrainStatusRecorderCallback(TrainerCallback):
    def __init__(self, trainer_ref, tokenized_train):
        self.trainer_ref = trainer_ref
        self.tokenized_train = tokenized_train
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_losses.append(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self.eval_losses.append(metrics["eval_loss"])

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()
    return {"eval_accuracy": accuracy.item()}

# ===== Data Processing =====
train_df = pd.read_excel("/src/dataset/train.xlsx").reset_index(drop=True)
val_df   = pd.read_excel("/src/dataset/val.xlsx").reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
eval_dataset  = Dataset.from_pandas(val_df, preserve_index=False)

# ===== Tokenizer =====
tokenizer = LlamaTokenizer.from_pretrained("/src")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    prompts = ["Give me a Flexscript. " + p for p in examples["Prompt"]]
    responses = examples["Response"]
    full_texts = [p + r for p, r in zip(prompts, responses)]
    tokenized = tokenizer(full_texts, padding="max_length", truncation=True, max_length=4096)
    labels = []
    for p, t in zip(prompts, tokenized["input_ids"]):
        prompt_len = len(tokenizer(p)["input_ids"])
        labels.append([-100] * prompt_len + t[prompt_len:])
    tokenized["labels"] = labels
    return tokenized

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["Prompt", "Response"])
tokenized_val   = eval_dataset.map(tokenize_function,   batched=True, remove_columns=["Prompt", "Response"])

# ===== Model =====
config = LlamaConfig.from_pretrained("/src")
model  = LlamaForCausalLM(config)
model.resize_token_embeddings(tokenizer.vocab_size)

# ===== Training =====
output_dir = "/results"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=2,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=200,
    fp16=True,
    report_to="none",
    gradient_checkpointing=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
)

status_callback = TrainStatusRecorderCallback(None, tokenized_train)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[GPUUsageCallback(), status_callback],
)
status_callback.trainer_ref = trainer

trainer.train()

# ===== Save Model =====
if torch.distributed.get_rank() == 0 or not torch.distributed.is_initialized():
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")
