import os
import copy
import logging
from utils import file_util
from collections import deque
from tqdm import tqdm

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

def trace_path(root_id, info, max_level):
    labels = {root_id: info["label"]}
    linkto_infos = cache_linkto(info["parents"])
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

def trace_parents(folder_name: str, start: int = 0, end: int = None, max_level: int = 2):
    file_names = file_util.get_file_name_in_dir(folder_name, "txt")
    labels_info, name_mention_info = {}, {}
    for file_name in tqdm(file_names[start:end], desc="Tracing parents"):
        base_name = os.path.splitext(file_name)[0]
        entities_file = base_name + "_entities.pck"
        for name_mention_id, info in file_util.load(entities_file).items():
            try:
                parents, labels = trace_path(name_mention_id, info, max_level=max_level)
            except KeyError:
                logging.debug(f"CANNOT TRACE {name_mention_id}")
                continue
            labels_info.update(labels)
            new_info = copy.deepcopy(info)
            new_info["parents"] = parents
            name_mention_info[name_mention_id] = new_info
            
    return labels_info, name_mention_info
