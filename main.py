import logging
import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter

from args import *
from data import *
from model import *
from train import *

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(module)s %(message)s", level=logging.DEBUG)
    args = make_run_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger = wandb.init(
        project="patent-graph", 
        entity="joanna_cin", 
        name=args.exp_name, 
        config=vars(args),
        sync_tensorboard=True,
    )

    dataset = prepare_dataset(
        args.data_path,
        args.cache_dir, 
        feature_type=args.feature_type.split(","), 
        lang=args.lang,
        par_num=list(map(int, args.par_num.split(","))), 
        n_clusters=args.n_clusters,
    )
    graph = dataset.get_graph()
    logging.info(graph)
    model = prepare_model(
        args.model_name, 
        graph,
        n_classes=dataset.num_classes,
        hidden_feat=args.hidden_feat, 
        n_layers=args.n_layers, 
        aggregate=args.aggregate, 
        num_heads=args.num_heads, 
        multihead_aggregate=args.multihead_aggregate, 
        dropout=args.dropout
    )
    model = run(model, graph, dataset, args.lr, args.epochs, args.threshold, args.strategy_name, writer=SummaryWriter(), exp_name=args.exp_name)
    logger.finish()
