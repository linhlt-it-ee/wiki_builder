from argparse import ArgumentParser


def make_data_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--data_path", type=str, help="Path to .ndjson file")
    parser.add_argument("--cache_dir", type=str, help="Data directory")
    parser.add_argument("--lang", type=str, default="en", help="Patent language")
    args = parser.parse_args()
    return args

def make_run_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--lang", type=str, default="en", help="Patent language")
    # graph args
    parser.add_argument("--data_path", type=str, help="Path to .ndjson file")
    parser.add_argument("--cache_dir", type=str, help="Data directory")
    parser.add_argument("--par_num", type=str, default="-1,-1,-1", help="Number of k-hop parents for each node")
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of `cluster` nodes")
    parser.add_argument("--feature_type", type=str, default="concept,word,cluster", help="Other types of nodes")
    parser.add_argument("--aggregate", type=str, default="sum", help="Aggregation function on neighboring nodes")
    parser.add_argument("--multihead_aggregate", type=str, default="concat", help="Aggregate function on attention heads")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    # model args
    parser.add_argument("--model_name", type=str, help="GNN architecture: `rgcn`, `rgcn2`, `rsage`, `rsage2`, `rgat`, `rgat2`")
    parser.add_argument("--hidden_feat", type=int, default=64, help="Number of hidden units")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    # training args
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")
    parser.add_argument("--strategy_name", type=str, default=None, help="Active learning query strategy, accepts `random`, `lc` (Least Confidence), `entropy`, `margin`, margin2`, and `kcenter`")
    args = parser.parse_args()
    return args
