import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, top_k_accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label_columns = [
    "lead",
    "tin",
    "retail",
    "fuel",
    "propane",
    "crude",
    "income",
    "oat",
    "copra-cake",
    "barley",
    "groundnut",
    "cotton-oil",
    "rand",
    "cpi",
    "lei",
    "cocoa",
    "groundnut-oil",
    "jobs",
    "nkr",
    "livestock",
    "castor-oil",
    "palmkernel",
    "money-fx",
    "sunseed",
    "hog",
    "nat-gas",
    "zinc",
    "coconut-oil",
    "gas",
    "rape-oil",
    "gold",
    "orange",
    "pet-chem",
    "wheat",
    "nickel",
    "jet",
    "interest",
    "carcass",
    "bop",
    "l-cattle",
    "potato",
    "rapeseed",
    "sugar",
    "coffee",
    "soy-oil",
    "money-supply",
    "platinum",
    "yen",
    "wpi",
    "ship",
    "soybean",
    "sorghum",
    "lin-oil",
    "dmk",
    "meal-feed",
    "coconut",
    "rice",
    "dlr",
    "alum",
    "oilseed",
    "acq",
    "reserves",
    "ipi",
    "corn",
    "grain",
    "housing",
    "nzdlr",
    "naphtha",
    "strategic-metal",
    "palm-oil",
    "sun-meal",
    "lumber",
    "tea",
    "rye",
    "rubber",
    "gnp",
    "veg-oil",
    "cpu",
    "silver",
    "copper",
    "soy-meal",
    "earn",
    "sun-oil",
    "instal-debt",
    "cotton",
    "heat",
    "trade",
    "dfl",
    "palladium",
    "iron-steel",
]


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


def get_text_embedding(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int = 64,
    experiment_type: str = "pool_first",
):
    encoded_input = tokenizer.encode_plus(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
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


def get_onehot_embedding(data):
    num_samples = len(data)
    encodings = []
    for i in tqdm(range(num_samples), desc="One-hot Encoding"):
        encoding = [0] * num_samples
        encoding[i] = 1
        encodings.append(encoding)
    return encodings


def compute_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, average="micro")
    auc = roc_auc_score(y_true, y_score)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc auc": auc}
