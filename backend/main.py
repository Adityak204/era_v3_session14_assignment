from fastapi import FastAPI
from pydantic import BaseModel
import torch
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.model import SmolLM

app = FastAPI()


@dataclass
class MainConfig:
    vocab_size: int = 49152
    emb_dim: int = 576
    intermediate_size: int = 1536
    num_layers: int = 30
    n_q_heads: int = 9
    n_kv_heads: int = 3
    max_seq_len: int = 1024
    dropout: float = 0.1
    rms_norm_eps: float = 1e-05
    init_std: float = 0.041666666666666664


# Initialize Model Configuration
config = MainConfig()
model = SmolLM(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Checkpoint
model_repo = "Adityak204/SmolLM2-135-cosmopedia-10k"
model_filename = "smolLM-v2.pth"
checkpoint_path = hf_hub_download(repo_id=model_repo, filename=model_filename)

checkpoint = torch.load(checkpoint_path, map_location=device)["model_state_dict"]
model.load_state_dict(checkpoint)
model.eval()  # Put model in evaluation mode

# Load Tokenizer Once (Global)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer.pad_token = tokenizer.eos_token


class GenerateRequest(BaseModel):
    text: str
    seq_length: int


def greedy_decode(model, input_ids, max_length=100, tokenizer=None):
    """Greedy decoding function for text generation."""
    current_ids = input_ids

    with torch.no_grad():
        for _ in range(max_length - current_ids.shape[1]):
            outputs = model(current_ids)
            last_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(last_token_logits, dim=-1).unsqueeze(0)

            current_ids = torch.cat([current_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return current_ids


def generate_prediction(model, prompt, max_length=100):
    """Generate text prediction from the model."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = greedy_decode(
            model, input_ids, max_length=max_length, tokenizer=tokenizer
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return {"prediction": generated_text}


@app.get("/")
async def read_root():
    return {"Welcome": "to the SmolLM2 API!"}


@app.post("/predict")
async def predict(request: GenerateRequest):
    return generate_prediction(model, request.text, request.seq_length)
