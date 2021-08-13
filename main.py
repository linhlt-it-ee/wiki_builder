import logging
from args import *
from data import *
from model import *
from train import *

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(module)s %(message)s", level=logging.DEBUG)
    args = make_args()
    graph, doc_ids, concept_ids = prepare_graph(args.data_dir, [args.par1_num, args.par2_num, args.par3_num])
    remove_doc_ids = [x for x in doc_ids.values() if graph.in_degrees(x, etype="in") == 0]
    graph = dgl.remove_nodes(graph, remove_doc_ids, ntype="doc")

    target_node = "doc"
    n_classes = graph.nodes[target_node].data["label"].shape[1]
    model = prepare_model(args.model_name, graph, n_classes, args.hidden_feat, args.n_layers)
    train(model, graph, target_node, args.lr, args.epochs, args.threshold)
