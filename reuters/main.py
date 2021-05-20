import sys
sys.path.append("../")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
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
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data"
DOC_PATH = os.path.join(DATA_DIR, "reuters.csv")
DOC_MASK_PATH = os.path.join(DATA_DIR, "mask.pck")
ENTITY_LABEL_PATH = os.path.join(DATA_DIR, "entity_labels.json")
CONCEPT_GRAPH_DIR = os.path.join(DATA_DIR, "graphs")

pretrained_model_name = "distilbert-base-uncased"
model = AutoModel.from_pretrained(pretrained_model_name).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
def get_bert_embedding(text: str, max_length: int = 128, embedding_type: str = "pool_first"):
    global model, tokenizer
    encoded_input = tokenizer.encode_plus(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
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
doc_df = pd.read_csv(DOC_PATH, index_col="index")
label_columns = [x for x in doc_df.columns if x not in ("path", "topic", "subset", "content")]
doc_labels = doc_df[label_columns].values
doc_masks = file_util.load(DOC_MASK_PATH)

# Creating features for node document (bert embedding from document content)
doc_features_file = os.path.join(DATA_DIR, "cached_doc_features.pt")
if os.path.exists(doc_features_file):
    doc_features = torch.load(doc_features_file)
else:
    doc_features = {}       # All key must be string
    for doc_index, doc_info in tqdm(doc_df.iterrows(), total=len(doc_df), desc="Creating doc features"):
        doc_index = str(doc_index)
        content = doc_info["content"].split("\n")[0]
        doc_features[doc_index] = get_bert_embedding(content, max_length=64)
    torch.save(doc_features, doc_features_file)

# Creating features for node concept (bert embeeding from concept label)
# Note: only encoding nodes appearing on concept graph
concept_graph_files = file_util.get_file_name_in_dir(CONCEPT_GRAPH_DIR, "gz")
entity_labels = file_util.load_json(ENTITY_LABEL_PATH)
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
doc_encodes = {doc_name : i for i, doc_name in enumerate(doc_features.keys())}
entity_encodes = {ent_name : i for i, ent_name in enumerate(entity_features.keys())}
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
        assert doc_index in doc_encodes, f"Cannot found document {doc_index} information" 
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
    num_nodes_dict=num_nodes_dict
)
graph.nodes["document"].data["feat"] = torch.tensor(list(doc_features.values()), dtype=torch.float32)
graph.nodes["document"].data["labels"] = torch.tensor(doc_labels, dtype=torch.long)
graph.nodes["concept"].data["feat"] = torch.tensor(list(entity_features.values()), dtype=torch.float32)
graph = graph.to(DEVICE)

print("\n****** Graph Information ******")
print(graph)
print("****** Label Information ******")
print(torch.sum(graph.nodes["document"].data["labels"], axis=0).detach().cpu())

# Creating model for node classification
num_train_epochs = 50000
threshold = 0.5
lr = 0.005
hid_dim = 768
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
    num_labels = y_true.shape[1]
    """
    thresholds = [find_optimal_threshold(y_true[:, i], y_prob[:, i]) for i in range(num_labels)]
    y_pred = np.zeros(y_prob.shape)
    for i in range(num_labels):
        y_pred[:, i] = (y_prob[:, i] >= thresholds[i]).astype(int)
    """
    y_pred = np.vectorize(lambda p : int(p > threshold))(y_prob)
    metrics = {}
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for reduce_type in ("micro avg", "macro avg"):
        for metric, score in clf_report[reduce_type].items():
            metrics[f"{reduce_type}_{metric}"] = score
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    # metrics["macro avg_auc"] = roc_auc_score(y_true, y_prob, average="macro")
    # metrics["micro avg_auc"] = roc_auc_score(y_true, y_prob, average="micro")
    return metrics

print("\n****** Training *******")
pbar = trange(num_train_epochs, desc="Training")
experiment_name = f"0.7_threshold={threshold}_dim={hid_dim}_lr={lr}_epochs={num_train_epochs}"
logger = SummaryWriter(log_dir=f"runs/{experiment_name}")
train_mask = doc_masks["train_mask"]
test_mask = doc_masks["test_mask"]

for epoch in pbar:
    logits = model(graph, "document")
    train_logits = logits[train_mask]
    test_logits = logits[test_mask]
    optimizer.zero_grad()
    loss = criterion(train_logits, graph.nodes["document"].data["labels"][train_mask].type_as(train_logits)), 
    loss = loss[0]

    pbar.set_postfix(loss=loss.item())
    logger.add_scalar("loss", loss.item(), epoch)
    if (epoch + 1) % 100 == 0:
        probs = torch.sigmoid(test_logits)
        metric_scores = compute_metrics(graph.nodes["document"].data["labels"][test_mask], probs)
        for metric, score in metric_scores.items():
            logger.add_scalar(metric, score, epoch)

    loss.backward()
    optimizer.step()

print("Last epoch results:")
probs = torch.sigmoid(test_logits)
metric_scores = compute_metrics(graph.nodes["document"].data["labels"][test_mask], probs)
for metric, score in metric_scores.items():
    print("- {}: {}".format(metric, score))

print("Last prediction maximum probabilities:")
res, _ = torch.max(probs, axis=0)
print(res)
