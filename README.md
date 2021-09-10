

## Preprocess Data

The data is organized in a single `.ndjson` file. Each line represents a `json` patent as follows:

| Attribute | Meaning | Datatype |
| - | - | - |
| id | Patent ID | str |
| desc | Patent Description | str |
| content | Patent Claims | str |
| labels | Patent Classification Codes | List[str] |
| is_train | Whether it is included for training | bool |
| is_val | Whether it is included for validation | bool |
| is_test | Whether it is included for testing | bool |

Sample processing notebook with English and Japanese patents: [US patents](./data/preprocess_us_patents.ipynb) | [JA patents](./data/preprocess_jp_patents.ipynb)


## 2. Train

The training pipeline includes: [prepare graph](#graph), [prepare model](#model), and [training](#train). To start training, modify `run.sh` with your data directory (as well as other settings in `args.py`) and use the following command:

```bash
bash run.sh
```

**Note**: We use WnB to monitor the experiments. You might need to either install and change the project name to your account in `main.py` or skip them in the code.

#### Graph

We use DGL to create a heterogeneous graph which can contain nodes and edges of multiple types.

| Node | Embedding Type |
| -- | -- |
| doc | BERT embedding (based on raw document's `desc`) |
| word | Word2Vec (extracted from raw document's `content`) |
| cluster | K-Means (based on doc's embedding) |
| concept | BERT embedding (based on concept's label) |

| Edge | Weight | Bi-directional |
| -- | -- | -- |
| (word, word, word#relate) | PMI | No |
| (doc, word, word#in) | TF-IDF | Yes |
| (cluster, cluster, cluster#relate) | Euclidean distance | No |
| (doc, cluster, cluster#form) | Euclidean distance (K-Means) | Yes |
| (concept, concept, concept#child) | - | No |
| (doc, concept, concept#have) | - | Yes |

**Note**:
- To represent bi-directional edges, we add two types of directed edges with opposite direction.
- `concept` nodes should be crawled from Wikidata prior to creating graph.
- Temporary data is cached in the subfolder `cache/` of the data directory. Users might need to clear the cached files to test with different data.

#### Model

Available architecture includes: R-GCN, R-GCN+Dense (RGCN2), R-SAGE, R-SAGE+Dense (RSAGE2), R-GAT, and R-GAT+Dense (RGAT2).

#### Train

We provide a training procedure with and without Active Learning strategies (i.e., Least Confidence, Entropy Sampling, Margin Sampling, K-centers Greedy Sampling). Users can turn on this mode by providing appropriate `strategy_name` in `run.sh`.

The testing results will be available as csv file in `./results/` once the training is done.

## Code Structure
```
.
├── args.py
├── data
│   └── prepare_graph.py
├── data_crawl
├── main.py
├── map_concepts.py
├── model
│   ├── prepare_model.py
│   └── ...
├── retrieve_concepts.py
├── run.sh
├── train
│   ├── metrics.py
│   ├── strategy.py
│   └── train.py
└── utils
```