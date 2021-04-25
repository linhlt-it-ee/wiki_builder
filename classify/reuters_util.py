from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label_columns = ['lead', 'tin', 'retail', 'fuel', 'propane', 'crude', 'income', 'oat', 'copra-cake', 'barley', 'groundnut', 'cotton-oil', 'rand', 'cpi', 'lei', 'cocoa', 'groundnut-oil', 'jobs', 'nkr', 'livestock', 'castor-oil', 'palmkernel', 'money-fx', 'sunseed', 'hog', 'nat-gas', 'zinc', 'coconut-oil', 'gas', 'rape-oil', 'gold', 'orange', 'pet-chem', 'wheat', 'nickel', 'jet', 'interest', 'carcass', 'bop', 'l-cattle', 'potato', 'rapeseed', 'sugar', 'coffee', 'soy-oil', 'money-supply', 'platinum', 'yen', 'wpi', 'ship', 'soybean', 'sorghum', 'lin-oil', 'dmk', 'meal-feed', 'coconut', 'rice', 'dlr', 'alum', 'oilseed', 'acq', 'reserves', 'ipi', 'corn', 'grain', 'housing', 'nzdlr', 'naphtha', 'strategic-metal', 'palm-oil', 'sun-meal', 'lumber', 'tea', 'rye', 'rubber', 'gnp', 'veg-oil', 'cpu', 'silver', 'copper', 'soy-meal', 'earn', 'sun-oil', 'instal-debt', 'cotton', 'heat', 'trade', 'dfl', 'palladium', 'iron-steel']

def get_onehot_label(row):
    res = [None] * len(label_columns)
    for i, label in enumerate(label_columns):
        res[i] = row[label]
    return res


def normalize(text: str):
    split_text = text.split("\n")
    subject = split_text.pop(0).strip()
    content = " ".join(x.strip() for x in split_text).strip()
    merge_text = "{} | {}".format(subject, content)
    return merge_text

def get_text_embedding(text: str, tokenizer: AutoTokenizer, model: AutoModel, max_length: int = 64, experiment_type: str = "pool_first"):
    encoded_input = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    hidden_state = model(**encoded_input).last_hidden_state[0]  # Batch: 1 sample
    if experiment_type == "pool_first":
        embedding = hidden_state[0]
    elif experiment_type == "pool_mean":
        embedding = torch.mean(hidden_state, axis=0)
    elif experiment_type == "flatten":
        embedding = torch.flatten(hidden_state)
    else:
        raise NotImplementedError
    embedding = embedding.detach().cpu().numpy()
    return embedding