import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import os, json, torch, gc
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    BitsAndBytesConfig,
)

# ====== settings ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
llm_path = "configs/TinyLlama-1.1b"

data_path = "dataset/train.jsonl"
image_dir = "dataset/images"
save_dir = "./OpenCLIP_Linear_Projection_Tinyllama"

batch_size = 2
epochs = 10
lr = 1e-5
max_len = 4096
gradient_accumulation_steps = 4

# ====== Dataset ======
class VLMFinetuneDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, processor, tokenizer):
        self.samples = []
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        obj = self.samples[idx]
        image_name = os.path.basename(obj["image"])
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        prompt = "Give me a Flexscript. " + obj["prompt"].strip()
        response = obj["response"].strip()

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=2048).input_ids
        response_ids = self.tokenizer(response, add_special_tokens=False, truncation=True, max_length=2048).input_ids

        input_ids = (prompt_ids + response_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_len]
        pad_len = max_len - len(input_ids)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels    += [-100] * pad_len
        attention_mask = [1] * (max_len - pad_len) + [0] * pad_len

        return pixel_values, torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)

# ====== Load：OpenCLIP + TinyLLaMA ======
print("Loading models...")
vision_encoder = CLIPVisionModel.from_pretrained(vision_path).to(device)
processor      = CLIPImageProcessor.from_pretrained(vision_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
llm = AutoModelForCausalLM.from_pretrained(
    llm_path,
    quantization_config=bnb_config,
    device_map={"": 0},
    local_files_only=True,
)
llm.eval()
for p in llm.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if getattr(llm.config, "pad_token_id", None) is None:
    llm.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"

# ====== Gradient checkpointing（only vision encoder） ======
if hasattr(vision_encoder, 'gradient_checkpointing_enable'):
    vision_encoder.gradient_checkpointing_enable()

# ====== Linear Projection Connector ======
class LinearProjectionConnector(nn.Module):
    def __init__(self, vision_hidden_size, llm_hidden_size):
        super().__init__()
        self.linear_proj = nn.Linear(vision_hidden_size, llm_hidden_size)
        nn.init.normal_(self.linear_proj.weight, std=0.01)
        nn.init.zeros_(self.linear_proj.bias)

    def forward(self, vision_features):
        # vision_features: [B, num_patches, vision_hidden_size]
        return self.linear_proj(vision_features)

proj = LinearProjectionConnector(
    vision_hidden_size=vision_encoder.config.hidden_size,
    llm_hidden_size=llm.get_input_embeddings().embedding_dim
).to(device)

# ====== Optimizer ======
params = list(vision_encoder.parameters()) + list(proj.parameters())
optimizer = torch.optim.AdamW(params, lr=lr)

# ====== Data and Scheduler ======
dataset = VLMFinetuneDataset(data_path, image_dir, processor, tokenizer)
train_size = int(0.9 * len(dataset))
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size)

lr_scheduler = get_scheduler(
    "cosine", optimizer=optimizer, num_warmup_steps=50,
    num_training_steps=(epochs * len(train_loader)) // gradient_accumulation_steps
)

# ====== Train ======
os.makedirs(save_dir, exist_ok=True)
train_losses, val_losses = [], []
best_val_loss = float("inf")

print("Start Training")
for epoch in range(epochs):
    vision_encoder.train(); proj.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (pixels, input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        pixels = pixels.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vision_out = vision_encoder(pixel_values=pixels)
            proj_embeds = proj(vision_out.last_hidden_state)  # [B, num_patches, D_llm]

            text_embeds = llm.get_input_embeddings()(input_ids)  # [B, T, D_llm]

            full_embeds = torch.cat([proj_embeds, text_embeds], dim=1)

            B = pixels.size(0)
            q_mask = torch.ones((B, proj_embeds.size(1)), dtype=torch.long).to(device)
            full_mask = torch.cat([q_mask, attention_mask], dim=1)

            ignore_labels = torch.full((B, proj_embeds.size(1)), -100, dtype=torch.long).to(device)
            full_labels = torch.cat([ignore_labels, labels], dim=1)

            outputs = llm(inputs_embeds=full_embeds, attention_mask=full_mask, labels=full_labels)
            loss = outputs.loss / gradient_accumulation_steps

        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * gradient_accumulation_steps

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ====== Val ======
    vision_encoder.eval(); proj.eval()
    val_loss = 0.0
    with torch.no_grad():
        for pixels, input_ids, attention_mask, labels in val_loader:
            pixels = pixels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                vision_out = vision_encoder(pixel_values=pixels)
                proj_embeds = proj(vision_out.last_hidden_state)
                text_embeds = llm.get_input_embeddings()(input_ids)
                full_embeds = torch.cat([proj_embeds, text_embeds], dim=1)

                B = pixels.size(0)
                q_mask = torch.ones((B, proj_embeds.size(1)), dtype=torch.long).to(device)
                full_mask = torch.cat([q_mask, attention_mask], dim=1)

                ignore_labels = torch.full((B, proj_embeds.size(1)), -100, dtype=torch.long).to(device)
                full_labels = torch.cat([ignore_labels, labels], dim=1)

                outputs = llm(inputs_embeds=full_embeds, attention_mask=full_mask, labels=full_labels)
                val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "vision_encoder": vision_encoder.state_dict(),
            "proj": proj.state_dict()
        }, os.path.join(save_dir, "best_model.pt"))
        print("Saved best model")

    gc.collect(); torch.cuda.empty_cache()
