import torch
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer

def get_text_embedding(model: AutoModel, tokenizer: AutoTokenizer, text: str, max_length: int = 64, kind: str = "pool_first", device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoded_input = tokenizer.encode_plus(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)
    hidden_state = model(**encoded_input).last_hidden_state[0]
    if kind == "pool_first":
        embed = hidden_state[0]
    elif kind == "pool_mean":
        embed = torch.mean(hidden_state, axis=0)
    elif kind == "flatten":
        embed = torch.flatten(hidden_state)
    else:
        raise NotImplementedError
    return embed.detach().cpu().numpy()

def get_onehot_encoding(labels: List[str], label_ids: Dict[str, int]):
    embed = [0] * len(label_ids)
    for x in labels:
        embed[label_ids[x]] = 1
    return embed