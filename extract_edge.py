import sys
import re
import os

sys.path.append("../")

from utils import file_util
from tqdm import tqdm
from typing import Dict, Any

max_dist = 5
data_dir = "./data"
name_mention_dir = os.path.join(data_dir, "reuters_entities")
name_mention_paths = file_util.get_file_name_in_dir_regex(
    name_mention_dir, "_entity_dict_iteration.pck"
)
links = file_util.load_json(os.path.join(data_dir, "reuters_all_entity.json"))


def crawl_parent(doc_id: int, curr_id: int, links: Dict[int, Any], level: int, max_level: int) -> None:
    global doc_vs_concepts_edges
    if level > max_level:
        return
    if level != 0:
        doc_vs_concepts_edges[doc_id][curr_id] = level
    try:
        for parent in links[curr_id]:
            crawl_parent(doc_id, parent, links, level + 1, max_level)
    except:
        pass


doc_vs_concepts_edges = {}
for file_path in tqdm(
    name_mention_paths, total=len(name_mention_paths), desc="Creating edges"
):
    file_name = os.path.basename(file_path)
    doc_id_pattern = re.match(
        r"name_mention_(\d+).txt_entity_dict_iteration.pck", file_name
    )
    assert doc_id_pattern, f"Cannot file document ID from {file_name}"
    doc_id = doc_id_pattern.groups()[0]
    doc_vs_concepts_edges[doc_id] = {}
    doc_entities = file_util.load(file_path).keys()

    for entity_id in doc_entities:
        crawl_parent(doc_id, entity_id, links, 0, max_dist)

file_util.dump_json(doc_vs_concepts_edges, os.path.join(data_dir, "doc_vs_concepts.json"))
