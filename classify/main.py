import sys
sys.path.append("../")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import networkx as nx
from utils import file_util
import pandas as pd
from tqdm import tqdm
import reuters_util
import dgl
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "../sample_data"
pretrained_model_name = "distilbert-base-uncased"
doc_features_file = os.path.join(data_dir, "cached_doc_features.pck")
concept_infor = file_util.load_json(os.path.join(data_dir, "reuters_all_entity_brief.json"))
doc_vs_concepts = file_util.load_json(os.path.join(data_dir, "doc_vs_concepts.json"))

# --------------------------------- Data Loading --------------------------------
df = pd.read_csv(os.path.join(data_dir, "reuters.csv"))
df["content"] = df["content"].apply(reuters_util.normalize)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name).to(device)

# --------------------------------- Graph nodes --------------------------------
encoded_doc_ids = {}
doc_features = []
doc_labels = []

got_features = False
if os.path.exists(doc_features_file):
    got_features = True
    doc_features = file_util.load(doc_features_file)

for _, doc in tqdm(df.iterrows(), total=len(df), desc="Creating node:document metadata"):
    doc_id = str(doc["index"])
    doc_content = doc["content"]
    doc_label = reuters_util.get_onehot_label(doc)
    encoded_doc_ids[doc_id] = len(encoded_doc_ids)
    doc_labels.append(doc_label)
    if not got_features:
        doc_features.append(reuters_util.get_text_embedding(doc_content, tokenizer, model))

if not got_features:
    file_util.dump(doc_features, doc_features_file)
doc_features = torch.tensor(doc_features, dtype=torch.float32, device=device)
doc_labels = torch.tensor(doc_labels, dtype=torch.long, device=device)

# --------------------------------- Graph edges --------------------------------
encoded_concept_ids = {}
concept_desc = []
edges = []
for doc_id in tqdm(doc_vs_concepts, total=len(doc_vs_concepts), desc="Create edges & node:concept metadata"):
    concepts = doc_vs_concepts[doc_id]
    for concept_id in concepts:
        if concept_id not in encoded_concept_ids:
            encoded_concept_ids[concept_id] = len(encoded_concept_ids)
            concept_desc.append(concept_infor[concept_id]["label"])
        dist = concepts[concept_id]
        encoded_doc_id = encoded_doc_ids[doc_id]
        encoded_concept_id = encoded_concept_ids[concept_id]
        edges.append((encoded_doc_id, encoded_concept_id))

# --------------------------------- Graph --------------------------------
num_nodes_dict = {"document": len(encoded_doc_ids), "concept": len(encoded_concept_ids)}
graph = dgl.heterograph(data_dict={("document", "has", "concept"): edges}, num_nodes_dict=num_nodes_dict).to(device)
graph.nodes["document"].data["features"] = doc_features
graph.nodes["document"].data["labels"] = doc_labels
