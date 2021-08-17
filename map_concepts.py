import os
import logging
from collections import deque
from tqdm import tqdm

import utils
from args import *


def map_concepts(data_dir: str):
    cache_dir = os.path.join(data_dir, "cache")
    mention_dir = os.path.join(cache_dir, "mentions")
    # saved path
    concept_path = os.path.join(cache_dir, "concept_labels.json")
    trace_concepts(mention_dir, concept_path)

def trace_concepts(mention_dir, concept_path):
    mention_paths = utils.get_file_name_in_dir(mention_dir, "txt")
    concepts = {}
    for fname in tqdm(mention_paths, desc="Tracing concepts"):
        basename = os.path.splitext(fname)[0]
        parent_path = basename + "_parents.json"
        concept_info = utils.load(basename + "_entities.pck")
        parents = {}
        for cid, info in concept_info.items():
            try:
                parents[cid], label = trace_path(cid, info, max_level=3)
            except KeyError:
                logging.info(f"Cannot trace {cid}")
            concepts.update(label)
    utils.dump_json(parents, parent_path)
    return concepts

def cache_linkto(parent_links):
    linkto_infos = {}
    for link in parent_links:
        src_id, src_label, dest_id = link["id"], link["label"], link["link_to"]
        if src_id not in linkto_infos:
            linkto_infos[src_id] = {"label": src_label, "link_to": set()}
        if dest_id != "":
            linkto_infos[src_id]["link_to"].add(dest_id)
    for src_id in linkto_infos:
        linkto_infos[src_id]["link_to"] = list(linkto_infos[src_id]["link_to"])

    return linkto_infos

def trace_path(root_id, entity, max_level):
    linkto_infos = cache_linkto(entity["parents"])
    labels = {root_id: entity["label"]}
    parents = []
    
    # BFS to get parent up to max_level
    queue = deque()
    queue.append(root_id)
    level = {root_id : 0}
    path = {root_id: str(root_id)}
    while len(queue) != 0:
        u = queue.popleft()
        for v in linkto_infos[u]["link_to"]:
            if v not in level and v in linkto_infos:
                level[v] = level[u] + 1
                path[v] = "{} >> {}".format(path[u], v)
                labels[v] = linkto_infos[v]["label"]
                parents.append({
                    "id": v, 
                    "level": level[v], 
                    "label": labels[v],
                    "path": path[v],
                })
                if level[v] < max_level:
                    queue.append(v)
    
    return parents, labels

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(module)s %(message)s", level=logging.DEBUG)
    args = make_data_args()
    map_concepts(args.data_dir)
