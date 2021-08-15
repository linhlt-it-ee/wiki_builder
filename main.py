import logging
import torch
import torch.nn as nn
import wandb
from torch.utils.tensorboard import SummaryWriter

from args import *
from data import *
from model import *
from train import *

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(module)s %(message)s", level=logging.DEBUG)
    args = make_args()
    run = wandb.init(
        project="patent-graph", 
        entity="joanna_cin", 
        name=args.exp_name, 
        config=vars(args),
        sync_tensorboard=True,
    )

    target_node = "doc"
    graph, doc_ids, concept_ids, n_classes = prepare_graph(args.data_dir, [args.par1_num, args.par2_num, args.par3_num])
    model = prepare_model(args.model_name, graph, n_classes, args.hidden_feat, args.n_layers, args.aggregate, args.num_heads, args.multihead_aggregate)
    writer = SummaryWriter()
    train(model, graph, target_node, args.lr, args.epochs, args.threshold, writer)
    run.finish()
