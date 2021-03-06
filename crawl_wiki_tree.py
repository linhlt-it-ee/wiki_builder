import threading
import sys
import utils.text_util as text_util
import utils.file_util as file_util
import utils.wiki_util as wiki_util
import utils.graph_builder_from_wiki as wiki_graph_util
import utils.excel_tree_level_export as excel_tree_level_export
'''@HOW TO USE: search_wiki_with_threads_v3 function'''
'''@folder_name: folder of name mentions which are splitted into smaller-sized file'''
'''@start: start index of file in folder'''
'''@end: end index of file in folder'''
'''@iteration: iteration number for example if you want to find direct parent this number =1, for upper parent this number could be 2, 3'''
'''@@@WARNING: max-iteration should be only 2'''
#text_util.search_wiki_with_thread_by_version(sys.argv[1:][0],int(sys.argv[1:][1]),int(sys.argv[1:][2]),int(sys.argv[1:][3]))
def search_wiki_with_threads(folder_name,start, end, iteration):
    file_names = file_util.get_file_name_in_dir(folder_name, "txt")
    for i, file_name in enumerate(file_names[start:end]):
        thread1 = threading.Thread(target=search_wiki_with_forward_iteration,
                                   args=(file_name, file_name + "_entity_dict_iteration.pck",file_name + "_entity_dict_not_found_iteration.pck", iteration))
        thread1.start()
        print("THREAD", i, " START")

def update_entity_description_shortname(input_dict,all_entities_mention_dict):
    total = len(list(input_dict.keys()))
    dict_info = {}
    new_file = open("entity_brief.txt", "w")
    count = 0
    for entity_id in input_dict:
        # print("INFO",all_entities_mention_dict[entity_id])
        entity_label, entity_desc, entitiy_shortnames = wiki_util.get_entity_info_from_id(entity_id)
        dict_info[entity_id] = {"label": entity_label, "description": entity_desc, "short_names": entitiy_shortnames,"name_mention":all_entities_mention_dict[entity_id]["name_mention"] if entity_id in all_entities_mention_dict else []}
        new_file.write(entity_id + ';' + entity_label + ';' + entity_desc + ';' + str(entitiy_shortnames))
        print(count, '/', total)
        count += 1
    new_file.close()
    return dict_info

def update_wiki_info2dict(entity_id, info_dict):
    if entity_id not in info_dict:
        label, description, shortnames = wiki_util.get_entity_info_from_id(entity_id)
        info_dict[entity_id] = {"label": label, "description": description, "short_names": shortnames}

def update_other_children_nodes(parent_of_leaf_nodes,link_data_dict,all_entities_info_dict):
    for i, leaf_parent_id in enumerate(parent_of_leaf_nodes):
        print(i, '/', len(parent_of_leaf_nodes))
        child_ids, child_labels = wiki_util.get_entities_SPARQ_by_property(leaf_parent_id,'P279',is_forward=False)
        instance_of_ids, instance_of_labels=wiki_util.get_entities_SPARQ_by_property(leaf_parent_id,'P31',is_forward=False)
        parent_of_instances=link_data_dict[leaf_parent_id]
        for instance_of_id in instance_of_ids:
            link_data_dict[instance_of_id]=link_data_dict.get(instance_of_id,[])
            parent_of_instances=[x for x in parent_of_instances if x not in link_data_dict[instance_of_id]]
            link_data_dict[instance_of_id].extend(parent_of_instances)
            update_wiki_info2dict(instance_of_id, all_entities_info_dict)
        for child_id in child_ids:
            link_data_dict[child_id] = link_data_dict.get(child_id, [])
            if leaf_parent_id not in link_data_dict[child_id]:
                link_data_dict[child_id].append(leaf_parent_id)
            update_wiki_info2dict(child_id, all_entities_info_dict)


'''@HOW TO USE: update_entity_details function'''
'''@PURPOSE: update description for all entities (included n-level parents), get all parents of leaf nodes to find other children, instance_of, sub_part_of'''
'''@folder_name: folder where you cut all name_mentions into smaller_sized_files'''
'''@file_regex:dict_iteration.pck'''
'''@output_path:name_of_your_output_file'''
def update_entity_details(folder_name,file_regex,output_path):
    file_names = file_util.get_file_name_in_dir_regex(folder_name, file_regex)
    link_data = {}
    parent_of_leaf = []
    all_entities_from_mention={}
    for file_name in file_names:
        print("file_name", file_name)
        entity_dict = file_util.load(file_name)
        # print(entity_dict)
        for entity_id in entity_dict:
            all_entities_from_mention[entity_id]=entity_dict[entity_id]
            linkto_infos = entity_dict[entity_id]["parents"]
            for linkto_info in linkto_infos:
                source_id = linkto_info['id']
                dest_id = linkto_info['link_to']
                if source_id == entity_id:
                    parent_of_leaf.append(dest_id)
                else:
                    parent_of_leaf.append(source_id)
                    parent_of_leaf.append(dest_id)
                link_data[source_id] = link_data.get(source_id, [])
                link_data[dest_id] = link_data.get(dest_id, [])
                if dest_id not in link_data[source_id] and dest_id != '':
                    link_data[source_id].append(dest_id)
    file_util.dump(link_data, output_path+".pck")  # "iteration3_data_dumped.pck"
    file_util.dump(parent_of_leaf, output_path + "_parent_leaf.pck")
    file_util.dump_json(link_data,output_path+".json")
    des_short_name_dict=update_entity_description_shortname(link_data,all_entities_from_mention)
    file_util.dump_json(des_short_name_dict,output_path+"_brief.json")
    wiki_graph_util.convert_to_tree(link_data,des_short_name_dict)
    file_util.dump_json(all_entities_from_mention,output_path+"_patent_entity_relations.json")
    excel_tree_level_export.demo(file_util.load_json("all_entity_level.json"))
    # update_other_children_nodes(parent_of_leaf,link_data,brief_dict)
def search_wiki_with_forward_iteration(txt_file, output_entity_file, not_wiki_output, iter_num=3):
    # lines = open("not_in_wiki_token_1710.txt", 'r').readlines()
    lines = open(txt_file, 'r').readlines()
    not_found_entity=[]
    total = len(lines)
    entity_dict={}
    for i, line in enumerate(lines):
        word = text_util.remove_characters('\n','',line)
        word = text_util.remove_characters('-', ' ', word)
        singu_word=text_util.convert_plural_to_singular(word)
        entities = wiki_util.get_wiki_id_from_text(word,entity_dict,iter_num)
        if singu_word !=word:
            entities.extend(wiki_util.get_wiki_id_from_text(singu_word,entity_dict,iter_num))
        if len(entities)== 0:
            not_found_entity.append(word)
        file_util.dump(entity_dict, output_entity_file)
        file_util.dump(not_found_entity, not_wiki_output)
        print(i, '/', total, ')')#, word, '###', entities, '###'
    # file_util.dump(entity_dict, "entities_dict_wth_lvl.pck")
    file_util.dump(entity_dict, output_entity_file)
    file_util.dump(not_found_entity,not_wiki_output)

if __name__ == "__main__":
    choice=int(sys.argv[1:][0])
    if not choice:#choice=0 folder_name, start, end, iteration
        # python3 sony_patent_evaluation/test/crawl_wiki_tree.py 0 entity_folder_09122019 0 10 2
        search_wiki_with_threads(sys.argv[1:][1],int(sys.argv[1:][2]),int(sys.argv[1:][3]),int(sys.argv[1:][4]))
    elif choice==1:
        #python3 sony_patent_evaluation/test/crawl_wiki_tree.py 1 "entity_folder_09122019" "_dict_iteration.pck" "09_12_2019"
        update_entity_details(sys.argv[1:][1],sys.argv[1:][2],sys.argv[1:][3])
    else:
        #python3 sony_patent_evaluation/test/crawl_wiki_tree.py 1 "entity_folder_03122019" "_dict_iteration.pck" "07_12_2019"
        excel_tree_level_export.demo(file_util.load_json("all_entity_level.json"))
# update_entity_details("entity_folder_09122019", "_dict_iteration.pck", "09_12_2019")
