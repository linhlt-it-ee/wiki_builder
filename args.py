from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, help="Experiment name for logging")
    # graph args
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--par1_num", type=int, default=-1, help="Number of 1-hop parents for each node")
    parser.add_argument("--par2_num", type=int, default=-1, help="Number of 2-hop parents for each node")
    parser.add_argument("--par3_num", type=int, default=-1, help="Number of 3-hop parents for each node")
    parser.add_argument("--pretrained_node_encoder", type=str, default="distilbert-base-uncased", help="Name of pretrained model for node embedding")
    # model args
    parser.add_argument("--model_name", type=str, help="GNN architecture")
    parser.add_argument("--hidden_feat", type=int, default=128, help="Number of hidden units")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of hidden layers")
    # training args
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")
    args = parser.parse_args()
    return args
