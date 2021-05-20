import sys

sys.path.append("../")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import json
import torch.optim as optim
import torch.nn as nn
import dgl
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm, trange
from utils import file_util
from heterorgcn import HeteroRGCN
from transformers import AutoModel, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    roc_curve,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data/google_patents/us-25000"
DOC_DIR = os.path.join(DATA_DIR, "doc")
ENTITY_LABEL_PATH = os.path.join(DATA_DIR, "entity_labels.json")
CONCEPT_GRAPH_DIR = os.path.join(DATA_DIR, "graphs")
DOC_MASK_PATH = os.path.join(DATA_DIR, "mask.pck")

pretrained_model_name = "distilbert-base-uncased"
model = AutoModel.from_pretrained(pretrained_model_name).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)


def get_bert_embedding(
    text: str, max_length: int = 128, embedding_type: str = "pool_first"
):
    global model, tokenizer
    encoded_input = tokenizer.encode_plus(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(DEVICE)
    hidden_state = model(**encoded_input).last_hidden_state[0]
    if embedding_type == "pool_first":
        embedding = hidden_state[0]
    elif embedding_type == "pool_mean":
        embedding = torch.mean(hidden_state, axis=0)
    elif embedding_type == "flatten":
        embedding = torch.flatten(hidden_state)
    else:
        raise NotImplementedError
    return embedding.detach().cpu().numpy()


# Getting features and labels encoding
# Creating features for node document (bert embedding from document content)
doc_files = file_util.get_file_name_in_dir(DOC_DIR, "json")
doc_info_file = os.path.join(DATA_DIR, "cached_doc_info.pck")
if os.path.exists(doc_info_file):
    doc_info = file_util.load(doc_info_file)
    doc_features = doc_info.pop("feat")
    doc_labels = doc_info.pop("labels")
    doc_names = doc_info.pop("doc_id")
    doc_label_mapping = doc_info.pop("label_mapping")
else:
    doc_types = set()
    doc_cnt = 0
    for doc_file in tqdm(doc_files, desc="Getting all document classifications"):
        with open(doc_file, "r") as f:
            for line in f:
                doc_cnt += 1
                doc = json.loads(line)
                doc_types.update([x["code"] for x in doc["classifications"]])

    print("Number of document types:", len(doc_types))
    doc_names, doc_features, doc_labels = [], [], []
    doc_label_mapping = {x: i for i, x in enumerate(doc_types)}
    pbar = trange(doc_cnt, desc="Getting doc features & labels")
    for doc_file in doc_files:
        with open(doc_file, "r") as f:
            for line in f:
                doc = json.loads(line)
                doc_index = doc["patent_id"]
                doc_names.append(doc_index)
                try:
                    doc_title = doc["title"][0]["text"]
                except IndexError:
                    doc_title = ""
                    print(f"Doc {doc_index} has no title")
                doc_features.append(get_bert_embedding(doc_title))
                label = [0] * len(doc_label_mapping)
                for x in doc["classifications"]:
                    label[doc_label_mapping[x["code"]]] = 1
                doc_labels.append(label)
                pbar.update(1)

    doc_info = {
        "doc_id": doc_names,
        "feat": doc_features,
        "labels": doc_labels,
        "label_mapping": doc_label_mapping,
    }
    file_util.dump(doc_info, doc_info_file)


# Creating features for node concept (bert embeeding from concept label)
# Note: only encoding nodes appearing on concept graph
concept_graph_files = file_util.get_file_name_in_dir(CONCEPT_GRAPH_DIR, "gz")
entity_labels = file_util.load_json(ENTITY_LABEL_PATH)
doc_mask = file_util.load(DOC_MASK_PATH)
train_mask, test_mask = doc_mask["train_mask"], doc_mask["test_mask"]
entity_features_file = os.path.join(DATA_DIR, "cached_entity_features.pt")
if os.path.exists(entity_features_file):
    entity_features = torch.load(entity_features_file)
else:
    entity_list = set()
    for file_name in tqdm(concept_graph_files, desc="Getting entity nodes"):
        graph = nx.read_edgelist(path=file_name)
        for u in graph.nodes:
            entity_list.add(u)

    entity_features = {}
    for eid, label in tqdm(entity_labels.items(), desc="Creating entity features"):
        if eid not in entity_list:
            continue
        entity_features[eid] = get_bert_embedding(label, max_length=32)
    torch.save(entity_features, entity_features_file)

# Getting edges
doc_encodes = {doc_name: i for i, doc_name in enumerate(doc_names)}
entity_encodes = {ent_name: i for i, ent_name in enumerate(entity_features.keys())}
DvsC_edgelist = []
CvsC_edgelist = []
for file_name in tqdm(concept_graph_files, desc="Collecting edge lists"):
    doc_index = os.path.basename(file_name).split("_")[0]
    graph = nx.read_edgelist(path=file_name)
    for e in graph.edges:
        u, v = e
        assert u in entity_encodes, f"Cannot found entity {u} information"
        assert v in entity_encodes, f"Cannot found entity {v} information"
        u_id, v_id = entity_encodes[u], entity_encodes[v]
        CvsC_edgelist.append((u_id, v_id))

    concepts = graph.nodes
    for c in concepts:
        assert (
            doc_index in doc_encodes
        ), f"Cannot found document {doc_index} information"
        assert u in entity_encodes, f"Cannot found entity {v} information"
        d_id, c_id = doc_encodes[doc_index], entity_encodes[c]
        DvsC_edgelist.append((d_id, c_id))

print("Number of DocVsConcept edges:", len(DvsC_edgelist))
print("Number of ConceptVsConcept edges:", len(CvsC_edgelist))

# Creating graph from nodes and edges
num_nodes_dict = {"document": len(doc_encodes), "concept": len(entity_encodes)}
graph = dgl.heterograph(
    data_dict={
        ("document", "have", "concept"): DvsC_edgelist,
        ("concept", "belong", "document"): [(v, u) for u, v in DvsC_edgelist],
        ("concept", "is_child", "concept"): CvsC_edgelist,
        ("concept", "is_parent", "concept"): [(v, u) for u, v in CvsC_edgelist],
    },
    num_nodes_dict=num_nodes_dict,
)
graph.nodes["document"].data["feat"] = torch.tensor(doc_features, dtype=torch.float32)
graph.nodes["document"].data["labels"] = torch.tensor(doc_labels, dtype=torch.long)
graph.nodes["concept"].data["feat"] = torch.tensor(
    list(entity_features.values()), dtype=torch.float32
)
graph = graph.to(DEVICE)

print("\n****** Graph Information ******")
print(graph)
print("****** Label Information ******")
print(torch.sum(graph.nodes["document"].data["labels"], axis=0).detach().cpu())

# Creating model for node classification
num_train_epochs = 2000
threshold = 0.55
lr = 0.001
hid_dim = 64
model_dims = {"in_dims": {}, "hid_dims": {}, "out_dims": {}}
out_dim = graph.nodes["document"].data["labels"].shape[1]
for ntype in graph.ntypes:
    model_dims["in_dims"][ntype] = graph.ndata["feat"][ntype].shape[1]
    model_dims["hid_dims"][ntype] = hid_dim
    model_dims["out_dims"][ntype] = out_dim

model = HeteroRGCN(graph, **model_dims)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()


def compute_metrics(y_true, y_prob):
    global threshold

    def find_optimal_threshold(y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return thresholds[np.argmax(tpr - fpr)]

    y_true = y_true.detach().cpu().numpy()
    y_prob = y_prob.detach().cpu().numpy()
    """
    num_labels = y_true.shape[1]
    thresholds = [find_optimal_threshold(y_true[:, i], y_prob[:, i]) for i in range(num_labels)]
    y_pred = np.zeros(y_prob.shape)
    for i in range(num_labels):
        y_pred[:, i] = (y_prob[:, i] >= thresholds[i]).astype(int)
    """
    y_pred = np.vectorize(lambda p: int(p > threshold))(y_prob)
    metrics = {}
    clf_report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    for reduce_type in ("micro avg", "macro avg"):
        for metric, score in clf_report[reduce_type].items():
            metrics[f"{reduce_type}_{metric}"] = score
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro avg_auc"] = roc_auc_score(y_true, y_prob, average="macro")
    metrics["micro avg_auc"] = roc_auc_score(y_true, y_prob, average="micro")
    return metrics


print("\n****** Training *******")
pbar = trange(num_train_epochs, desc="Training")
experiment_name = (
    f"threshold={threshold}_dim={hid_dim}_lr={lr}_epochs={num_train_epochs}"
)
logger = SummaryWriter(log_dir=f"runs/{experiment_name}")

for epoch in pbar:
    logits = model(graph, "document")
    train_logits = logits[train_mask]
    test_logits = logits[test_mask]
    optimizer.zero_grad()
    loss = (
        criterion(
            train_logits,
            graph.nodes["document"].data["labels"][train_mask].type_as(train_logits),
        ),
    )
    loss = loss[0]

    pbar.set_postfix(loss=loss.item())
    logger.add_scalar("loss", loss.item(), epoch)
    if (epoch + 1) % 100 == 0:
        probs = torch.sigmoid(test_logits)
        metric_scores = compute_metrics(
            graph.nodes["document"].data["labels"][test_mask], probs
        )
        for metric, score in metric_scores.items():
            logger.add_scalar(metric, score, epoch)
    loss.backward()
    optimizer.step()

print("Last epoch results:")
probs = torch.sigmoid(test_logits)
metric_scores = compute_metrics(
    graph.nodes["document"].data["labels"][test_mask], probs
)
for metric, score in metric_scores.items():
    print("- {}: {}".format(metric, score))
pd.DataFrame(metric_scores).to_csv("report.csv")

print("Last prediction maximum probabilities:")
res, _ = torch.max(probs, axis=0)
print(res)
