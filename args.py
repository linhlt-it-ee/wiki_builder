from argparse import ArgumentParser


def make_data_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--lang", type=str, default="en", help="Patent language")
    args = parser.parse_args()
    return args

def make_run_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, help="Experiment name for logging")
    # graph args
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--par1_num", type=int, default=-1, help="Number of 1-hop parents for each node")
    parser.add_argument("--par2_num", type=int, default=-1, help="Number of 2-hop parents for each node")
    parser.add_argument("--par3_num", type=int, default=-1, help="Number of 3-hop parents for each node")
    parser.add_argument("--text_encoder", type=str, default="distilbert-base-uncased", help="Strategy to generate embedding for text, accept pretrained HuggingFace model name")
    parser.add_argument("--aggregate", type=str, default="sum", help="Aggregation function on neighboring nodes")
    parser.add_argument("--multihead_aggregate", type=str, default="concat", help="Aggregate function on attention heads")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    # model args
    parser.add_argument("--model_name", type=str, help="GNN architecture")
    parser.add_argument("--hidden_feat", type=int, default=128, help="Number of hidden units")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    # training args
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")
    parser.add_argument("--strategy_name", type=str, default="lc", help="Active learning query strategy")
    args = parser.parse_args()
    return args
