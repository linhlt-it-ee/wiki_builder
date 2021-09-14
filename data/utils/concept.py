from collections import defaultdict
from typing import List, Dict

import networkx as nx

def create_document_concept_graph(doc_ids: List[str], doc_mention: Dict, mention_concept: Dict, par_num: List[int]):
    doc_ids = set(doc_ids)
    all_mentions = set([e for did, mention in doc_mention.items() if did in doc_ids for e in mention])

    # create concept graph
    CvsC_graph = nx.DiGraph()
    mention_ids = defaultdict(list)
    mentions = set()
    for mid, x in mention_concept.items():
        if len(x["parents"]) == 0:
            continue
        for label in x["name_mention"]:
            label = label.lower()
            if label in all_mentions:
                mention_ids[label].append(mid)
                mentions.add(mid)
        for par in x["parents"]:
            path = par["path"].split(" >> ")
            nx.add_path(CvsC_graph, path)
    print(f"Original concept graph: {CvsC_graph}")

    # prune concept graph
    C = set()
    children = mentions
    C.update(children)
    for level, parlevel_num in enumerate(par_num, start=1):
        cnt = defaultdict(lambda : 0)
        for child in children:
            for par in CvsC_graph.successors(child):
                if par not in C:    # disable nodes in previous levels
                    cnt[par] += 1
        # discard parents with less than 2 children
        children = [k for k, v in sorted(cnt.items(), key=lambda x : (x[1], x[0]), reverse=True) if v >= 2][:parlevel_num]
        C.update(children)
        print(f"Extracting {parlevel_num} parents level {level} with most children, got {len(children)}")
    C = C.difference(mentions)
    CvsC_graph = nx.DiGraph(CvsC_graph.subgraph(C))
    print(f"CvsC graph: {CvsC_graph}")
    print(f"C size: {len(C)}")
   
    DvsC_graph = nx.DiGraph()
    for did, mentions in doc_mention.items():
        if did not in doc_ids:
            continue
        DvsC_graph.add_node(did)
        # traverse meanings of a name_mention in a document
        for label in mentions:
            for mid in mention_ids[label]:
                # consider concept level 1 of each meaning
                for concept in mention_concept[mid]["parents"]:
                    if concept["level"] == 1 and concept["id"] in CvsC_graph.nodes:
                        DvsC_graph.add_edge(did, concept["id"])
    print(f"DvsC graph: {DvsC_graph}")

    return list(C), CvsC_graph.edges, DvsC_graph.edges