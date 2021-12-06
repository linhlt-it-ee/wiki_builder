import neptune.new as neptune
import numpy as np
import torch

from args import *
from data import *
from model import *
from train import *

if __name__ == "__main__":
    args = make_run_args()
    with open(f"api_token.txt", "r") as f:
        run = neptune.init(project=f"joanna/graph-patent", api_token=f.read().strip())
        run["params"] = vars(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = prepare_dataset(
        args.data_path,
        args.cache_dir,
        feature_type=args.feature_type.split(","),
        lang=args.lang,
        par_num=list(map(int, args.par_num.split(","))),
        n_clusters=args.n_clusters,
    )
    graph = dataset.get_graph()
    model = prepare_model(
        args.model_name,
        graph,
        n_classes=dataset.get_num_classes(),
        hidden_feat=args.hidden_feat,
        n_layers=args.n_layers,
        aggregate=args.aggregate,
        dropout=args.dropout,
        num_heads=args.num_heads,
        lang=args.lang,
    )
    model = train(
        model,
        graph,
        dataset,
        lr=args.lr,
        epochs=args.epochs,
        threshold=args.threshold,
        run=run,
    )
