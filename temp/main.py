import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import urllib.request

import dgl
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer

from heterorgcn import HeteroRGCN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiment_type = "flatten"
acm_data_url = "https://data.dgl.ai/dataset/ACM.mat"
acm_data_file_path = "/tmp/ACM.mat"
pretrained_model = "distilbert-base-uncased"
max_seq_len = 64
num_train_epochs = 10000
lr = 0.1

writer = SummaryWriter(os.path.join("runs", experiment_type))
print("Running type:", experiment_type, end="\n\n")

# --------------------------- Data Preparation ----------------------------------
urllib.request.urlretrieve(acm_data_url, acm_data_file_path)
acm_data = scipy.io.loadmat(acm_data_file_path)
abbr = {"paper": "P", "author": "A", "subject": "L"}
graph = dgl.heterograph(
    {
        ("paper", "written-by", "author"): acm_data["PvsA"].nonzero(),
        ("author", "writing", "paper"): acm_data["PvsA"].transpose().nonzero(),
        ("paper", "citing", "paper"): acm_data["PvsP"].nonzero(),
        ("paper", "cited-by", "paper"): acm_data["PvsP"].transpose().nonzero(),
        ("paper", "is-about", "subject"): acm_data["PvsL"].nonzero(),
        ("subject", "has", "paper"): acm_data["PvsL"].transpose().nonzero(),
    },
    device=device,
)
print("****** Graph Information ******")
print(graph, end="\n\n")

__, node_labels = acm_data["PvsC"].nonzero()
node_labels = torch.tensor(node_labels, dtype=torch.long, device=device)
num_labels = acm_data["C"].shape[0]

# --------------------------- Data Featurization ----------------------------------
def get_text_embedding(data):
    global experiment_type
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModel.from_pretrained(pretrained_model).to(device)
    embeddings = []
    for x in tqdm(data, total=len(data), desc="Embedding"):
        sample_input = tokenizer(
            x[0][0], padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt"
        )
        sample_input = {k: v.to(device) for k, v in sample_input.items()}
        sample_hidden_state = model(**sample_input).last_hidden_state[0]  # Batch: 1 sample
        if experiment_type == "pool_first":
            sample_embedding = sample_hidden_state[0]
        elif experiment_type == "pool_mean":
            sample_embedding = torch.mean(sample_hidden_state, axis=0)
        elif experiment_type == "flatten":
            sample_embedding = torch.flatten(sample_hidden_state)
        else:
            raise NotImplementedError
        sample_embedding = sample_embedding.detach().cpu().numpy()
        embeddings.append(sample_embedding)

    return embeddings


def get_onehot_encoding(data):
    num_samples = len(data)
    encodings = []
    for i in tqdm(range(num_samples), desc="One-hot Encoding"):
        encoding = [0] * num_samples
        encoding[i] = 1
        encodings.append(encoding)
    return encodings


# Node featurization
cached_node_features_file = f"/tmp/node_features_{experiment_type}.pt"
if os.path.exists(cached_node_features_file):
    node_features = torch.load(open(cached_node_features_file, "rb"))
else:
    node_features = {}
    for ntype in graph.ntypes:
        if ntype in ("paper"):
            node_features[ntype] = get_text_embedding(acm_data[abbr[ntype]])
        else:
            node_features[ntype] = get_onehot_encoding(acm_data[abbr[ntype]])
    torch.save(node_features, cached_node_features_file)

for ntype in node_features:
    node_features[ntype] = torch.tensor(node_features[ntype], dtype=torch.float32, device=device)

# Edge featurization
edge_type_ids = {etype: eid for eid, etype in enumerate(graph.etypes)}
edge_ids = {}
for etype in graph.etypes:
    edge_ids[etype] = torch.ones(graph.number_of_edges(etype), dtype=torch.long, device=device) * edge_type_ids[etype]

graph.ndata["feat"] = node_features
graph.edata["id"] = edge_ids

# --------------------------- Training ----------------------------------
model_dims = {"in_dims": {}, "hid_dims": {}, "out_dims": {}}
hid_dim = 32
out_dim = num_labels
for ntype in node_features:
    model_dims["in_dims"][ntype] = graph.ndata["feat"][ntype].shape[1]
    model_dims["hid_dims"][ntype] = hid_dim
    model_dims["out_dims"][ntype] = out_dim

model = HeteroRGCN(graph, **model_dims)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

print("****** Training ******")
pbar = trange(num_train_epochs, desc="Training")
for epoch in pbar:
    logits = model(graph, "paper")
    preds = torch.argmax(logits, axis=1)
    loss = criterion(logits, node_labels)
    writer.add_scalar("loss", loss.item(), epoch)
    pbar.set_postfix(loss=loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
