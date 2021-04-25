import json
from typing import List, Dict, Set
import copy
import utils.file_util as file_util
from tqdm import tqdm

class Level:
    def __init__(self):
        self.paths: List[List[str]] = []

    def add_parent(self, key):
        if self.paths:
            for path in self.paths:
                if key != path[-1]:
                    # if key != path[-1]:
                    path.append(key)
        else:
            self.paths.append([key])

    def merge(self, other):
        self.paths.extend(other.paths)

    def is_empty(self):
        return len(self.paths) == 0

    def copy(self):
        level = Level()
        level.paths = self.paths.copy()
        self.clean()
        return level

    def clean3(self):
        paths = []
        for s in {tuple(l) for l in self.paths}:
            paths.append(list(s))
        self.paths = paths

    def clean(self):
        def is_equal(a: List[str], b: List[str]):
            if len(a) != len(b):
                return False
            for k in range(len(a)):
                if a[k] != b[k]:
                    return False
            return True

        paths = []

        for i in range(len(self.paths)):
            is_duplicated = False
            for j in range(i + 1, len(self.paths)):
                if is_equal(self.paths[i], self.paths[j]):
                    is_duplicated = True
                    break
            if not is_duplicated:
                paths.append(self.paths[i])
        self.paths = paths


class Node:
    counter = 0

    def __init__(self, data: str):
        self.data = data
        self.parents: Set[Node] = set()
        self.children: Set[Node] = set()
        self.level: Level = None

    def add_parent(self, parent):
        self.parents.add(parent)
        parent.children.add(self)

    def add_parents(self, parents):
        for parent in parents:
            self.add_parent(parent)

    def build_level(self) -> Level:
        if self.level is not None:
            return copy.deepcopy(self.level)
        self.level = Level()
        Node.counter += 1
        for parent in self.parents:
            level = parent.build_level()
            level.add_parent(parent.data)
            self.level.merge(level)
        self.level.clean()
        return copy.deepcopy(self.level)


class Graph:
    counted_head=0
    counted_head_dict={}
    node_dict={}
    info_dict={}
    def __init__(self):
        self.nodes: Dict[str, Node] = dict()

    def grab_node(self, key: str):
        if key in self.nodes:
            return self.nodes[key]
        else:
            node = Node(key)
            self.nodes[key] = node
            return node

    def add(self, child: str, parents: List[str]):
        self.grab_node(child).add_parents([self.grab_node(p) for p in parents])

    @property
    def heads(self):
        return [node for node in self.nodes.values() if len(node.parents) == 0]

    def export_level(self):
        # with open('level2.json', 'w', encoding='utf8') as f:
        nodes = self.nodes.values()
        node_total=len(nodes)
        for i,node in enumerate(nodes):
            print("node:",i,'/',node_total)
            if node.data=="Q178674":
                print("PARENT",node.parents)
                # print("PATHS",node.level.paths)
            node_level_dict={}
            entity_id = node.data
            node.build_level()
            # text += json.dumps({node.data: list(node.level.paths)}) + '\n'
            # f.write(text)
            paths = node.level.paths
            if len(paths) > 0:
                # print("ERROR PATH",paths)
                for path in paths:
                    root_nodes =list( set([x for x in list(self.counted_head_dict.keys())]).intersection(set(path)))
                    # root_index = [i for i, x in enumerate(path) if x in root_nodes]

                    if len(root_nodes)>0:
                    # if "Q35120" in root_nodes:
                        # print("root_index",root_index)
                        root_node=root_nodes[0]
                        if root_node in self.counted_head_dict:
                            if root_node not in node_level_dict:
                                node_level_dict[root_node]={"level":0,"path":""}
                            if node_level_dict[root_node]['level']==0 or node_level_dict[root_node]['level']>len(path):
                                node_level_dict[root_node]['level'] = len(path)
                                node_level_dict[root_node]['path'] = ">>".join(path)

                            # self.node_dict[entity_id] = {"root_node": root_node, "paths": [],"level":node_level}
                            # for path in paths:
                            # path.append(entity_id)
                for root_node in node_level_dict:
                    self.counted_head_dict[root_node]['children'].append(
                        {'id': entity_id, 'level': node_level_dict[root_node]['level'],'path':node_level_dict[root_node]['path'], 'label': self.info_dict[entity_id]['label'],
                         'description': self.info_dict[entity_id]['description'],
                         'short_names': self.info_dict[entity_id]['short_names']})
                    self.counted_head_dict[root_node]['count'] = self.counted_head_dict[root_node]['count'] + 1

                # print('ROOT',list(node.parents)[0].data,list(node.parents)[0].data in self.counted_head_dict)
        with open('all_entity_level.json', 'w') as outfile:
            json.dump( self.counted_head_dict, outfile)

def merge_crawled_data(folder_name, file_type,output_path):
    # "id": parent_id, "label": parent_labels[ii], "link_to": parent_link_tos[ii]
    file_names = file_util.get_file_name_in_dir_regex(folder_name, file_type)
    data_dumped={}
    for file_name in tqdm(file_names, desc="Merge crawled data", total=len(file_names)):
        # print("file_name", file_name)
        entity_dict = file_util.load(file_name)
        # print(entity_dict)
        for entity_id in entity_dict:
            linkto_infos=entity_dict[entity_id]["parents"]
            for linkto_info in linkto_infos:
                source_id=str(linkto_info['id']).split('/')[-1]
                dest_id=linkto_info['link_to']
                data_dumped[source_id]=data_dumped.get(source_id,[])
                data_dumped[dest_id]=data_dumped.get(dest_id,[])
                if dest_id not in data_dumped[source_id] and dest_id!='':
                    data_dumped[source_id].append(dest_id)
    file_util.dump(data_dumped,output_path+'.pck')#"iteration3_data_dumped.pck"
    with open(output_path+'.json', 'w') as outfile:
        json.dump(data_dumped, outfile)

def convert_to_tree(link_dict,entity_info_dict):
    graph = Graph()
    # Load graph
    # with open(filename, 'rb') as f:
    for child, parent in link_dict.items():
        graph.add(child, parent)
    #info_dict=file_util.load("all_entities_info.pck")
    graph.info_dict = entity_info_dict
    heads = graph.heads
    print(f'Number of top-level node: {len(heads)}/{len(graph.nodes)}')
    count=0
    graph_counted_heads={}
    for head in heads:
        if len(head.children)>0:
            # print("head",len(head.children),head.data,entity_info_dict.get(head.data,""),)
            # for child in head.children:
                # print(child.data,entity_info_dict[child.data]['label'])
            count+=1
            graph_counted_heads[head.data]={'label':entity_info_dict[head.data]['label'],'count':0,'children':[]}

    print('----')
    graph.counted_head=count
    graph.counted_head_dict=graph_counted_heads
    print('TOTAL:',count)
    graph.export_level()

# convert_to_tree(file_util.load_json("/mnt/c/Cinnamon/12-12-2019/09_12_2019.json"),file_util.load_json("/mnt/c/Cinnamon/12-12-2019/09_12_2019_brief.json"))
# convert_to_tree(file_util.load_json("data_0412.json"), file_util.load("data_0412_brief.pck"))#link_data_dict
