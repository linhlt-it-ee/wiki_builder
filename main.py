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

    target_node = "doc"
    graph, doc_ids, n_classes = prepare_graph(args.data_dir, args.feature_type, [args.par1_num, args.par2_num, args.par3_num])
    model = prepare_model(args.model_name, graph, n_classes, args.hidden_feat, args.n_layers, args.aggregate, args.num_heads, args.multihead_aggregate, args.dropout)
    writer = SummaryWriter()
    model = run(model, graph, target_node, args.lr, args.epochs, args.threshold, args.strategy_name, writer, exp_name=args.exp_name)
    logger.finish()
