import os
import sys

sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
from utils import file_util

import reuters_util
from heterorgcn import HeteroRGCN
from han import HAN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

# Arguments
lr = 0.01
threshold = 0.5
num_train_epochs = 2000
weight_range = range(1, 3 + 1)
data_dir = "../data"
pretrained_model_name = "distilbert-base-uncased"
doc_features_file = os.path.join(data_dir, "cached_doc_features.pck")
concept_features_file = os.path.join(data_dir, "cached_concept_features.pck")
concept_infor = file_util.load_json(
    os.path.join(data_dir, "reuters_all_entity_brief.json")
)
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
encoded_concept_ids = {}
concept_features = []
edges = []

got_features = False
if os.path.exists(doc_features_file):
    got_features = True
    doc_features = file_util.load(doc_features_file)
    concept_features = file_util.load(concept_features_file)

for _, doc in tqdm(
    df.iterrows(), total=len(df), desc="Creating node:document metadata"
):
    doc_id = str(doc["index"])
    doc_content = doc["content"]
    doc_label = reuters_util.get_onehot_label(doc)
    encoded_doc_ids[doc_id] = len(encoded_doc_ids)
    doc_labels.append(doc_label)
    if not got_features:
        doc_features.append(
            reuters_util.get_text_embedding(
                doc_content, tokenizer, model, max_length=256
            )
        )

# --------------------------------- Graph edges --------------------------------
for doc_id in tqdm(
    doc_vs_concepts,
    total=len(doc_vs_concepts),
    desc="Create edges & node:concept metadata",
):
    doc_id = str(doc_id)
    concepts = doc_vs_concepts[doc_id]
    for concept_id in concepts:
        dist = concepts[concept_id]
        concept_label = concept_infor[concept_id]["label"] 
        if dist not in weight_range:
            continue
        if concept_id not in encoded_concept_ids:
            encoded_concept_ids[concept_id] = len(encoded_concept_ids)
            if not got_features:
                concept_features.append(reuters_util.get_text_embedding(
                    concept_label, tokenizer, model, max_length=64))
        encoded_doc_id = encoded_doc_ids[doc_id]
        encoded_concept_id = encoded_concept_ids[concept_id]
        edges.append((encoded_doc_id, encoded_concept_id, dist))
        

if not got_features:
    file_util.dump(doc_features, doc_features_file)
    file_util.dump(concept_features, concept_features_file)

doc_features = torch.tensor(doc_features, dtype=torch.float32, device=device)
doc_labels = torch.tensor(doc_labels, dtype=torch.long, device=device)
concept_features = torch.tensor(concept_features, dtype=torch.float32, device=device)

# --------------------------------- Graph --------------------------------
num_nodes_dict = {"document": len(encoded_doc_ids), "concept": len(encoded_concept_ids)}
graph = dgl.heterograph(
    data_dict={
        ("document", "have", "concept"): [(u, v) for u, v, w in edges],
        ("concept", "belong", "document"): [(v, u) for u, v, w in edges],
    },
    num_nodes_dict=num_nodes_dict,
).to(device)
print(graph)

graph.nodes["document"].data["feat"] = doc_features
graph.nodes["document"].data["label"] = doc_labels
graph.nodes["concept"].data["feat"] = concept_features
for etype_id, etype in enumerate(graph.etypes):
    graph.edges[etype].data["id"] = (
        torch.ones(graph.number_of_edges(etype), dtype=torch.long, device=device)
        * etype_id
    )
    graph.edges[etype].data["weight"] = torch.tensor(
        [w for u, v, w in edges], dtype=torch.float32, device=device
    )

# --------------------------- Training ----------------------------------
model_dims = {"in_dims": {}, "hid_dims": {}, "out_dims": {}}
hid_dim = 256
out_dim = len(reuters_util.label_columns)
for ntype in graph.ntypes:
    model_dims["in_dims"][ntype] = graph.ndata["feat"][ntype].shape[1]
    model_dims["hid_dims"][ntype] = hid_dim 
    model_dims["out_dims"][ntype] = out_dim
print("in_dims", model_dims["in_dims"], end="\n\n")

model = HeteroRGCN(graph, **model_dims)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

print("****** Training ******")
pbar = trange(num_train_epochs, desc="Training")
for epoch in pbar:
    logits = model(graph, "document")
    loss = criterion(logits, doc_labels.type_as(logits))

    probs = torch.sigmoid(logits).cpu().detach().numpy()
    preds = np.vectorize(lambda p: int(p >= threshold))(probs)
    scores = reuters_util.compute_metrics(
        doc_labels.cpu().detach().numpy(), preds, probs
    )
    pbar.set_postfix(loss=loss.item())
    writer.add_scalar("loss", loss.item(), epoch)
    for metric in scores:
        writer.add_scalar(metric, scores[metric], epoch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
