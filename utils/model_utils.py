import torch
from typing import List, Dict, Tuple
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_encoder(pretrained_model_name):
    model = AutoModel.from_pretrained(pretrained_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    return (model, tokenizer)

def get_text_embedding(encoder: Tuple[AutoModel, AutoTokenizer], text: str, max_length: int = 64, kind: str = "pool_first"):
    model, tokenizer = encoder
    inputs = tokenizer.encode_plus(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)
    hidden_state = model(**inputs).last_hidden_state[0]
    if kind == "pool_first":
        embedding = hidden_state[0]
    elif kind == "pool_mean":
        embedding = torch.mean(hidden_state, axis=0)
    elif kind == "flatten":
        embedding = torch.flatten(hidden_state)
    else:
        raise NotImplementedError
    return embedding

def get_onehot(labels: List[str], label_ids: Dict[str, int]):
    embed = [0] * len(label_ids)
    for x in labels:
        embed[label_ids[x]] = 1
    return embed
