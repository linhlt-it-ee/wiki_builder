from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

sparql = SPARQLWrapper("https://query.wikidata.org/sparql",agent="Mozilla/5.0")#headers={'User-Agent': 'Mozilla/5.0'}

def text2id(name_mention):
    query_string="""SELECT?item?label?tag?match WHERE{VALUES?tag{'"""+name_mention+"""' @en} VALUES ?match {rdfs:label skos:altLabel} ?item ?match ?tag. ?item rdfs:label ?label .FILTER(LANG(?label)="en")}"""
    try:
        sparql.setQuery(query_string)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        results_df = pd.io.json.json_normalize(results['results']['bindings'])
        items = list(results_df['item.value'].values) if 'item.value' in results_df else []
        items = [x.split('/')[-1] for x in items]
        values = list(results_df['label.value'].values) if 'item.value' in results_df else []
        ids = [x.split('/')[-1] for i, x in enumerate(items) if values[i] != x and type(values[i]) == str]
        labels = [x.lower() for i, x in enumerate(values) if x != items[i] and type(x) == str]
        return ids, labels
    except Exception:
        print("QUERY ERROR", name_mention)
        log_file = open("error_log.txt", 'a')
        log_file.write("get_entities_SPARQ_by_property_reverse" + name_mention+ '\n')
        log_file.close()
        # raise
        return [], []

def get_entity_info_from_id(entity_id):
    print(entity_id)
    query_string=""" SELECT
  ?id?label?desc
  (GROUP_CONCAT(DISTINCT(?aka); separator="|") AS ?akas)
WHERE{
  VALUES ?id { wd:"""+entity_id+""" }
  OPTIONAL{ ?id rdfs:label ?label. FILTER(LANG(?label)="en")}
  OPTIONAL{ ?id skos:altLabel ?aka . FILTER(LANG(?aka) = "en")}
  OPTIONAL{ ?id schema:description ?desc . FILTER(LANG(?desc) = "en")}
}
GROUP BY ?id?label?desc"""
    try:
        sparql.setQuery(query_string)
        # print(sparql.queryString)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # print(results)
        results_df = pd.io.json.json_normalize(results['results']['bindings'])
        # for item in results_df:
        #     print(item)
        # print(results_df[['item.value']['itemLabel.value']].heading())
        items = list(results_df['item.value'].values) if 'item.value' in results_df else []
        short_names = results_df['akas.value'].values if 'akas.value' in results_df else []
        short_names = [x for x in short_names if len(x) > 0]
        if len(short_names) > 0:
            short_names = short_names[0].split('|')
        description=results_df['desc.value'].values if 'desc.value' in results_df else []
        label = results_df['label.value'].values if 'label.value' in results_df else []
        # for value in values:
        #     print(type(value),value)
        # for value in items:
        #     print(type(value),value)

        # print("last_value",labels[-1])
        return label[0] if label else '',description[0] if description else '',short_names

    except Exception:
        print("GET INFO ERROR")
        log_file = open("error_log.txt", 'a')
        log_file.write("bottom_up_error" + entity_id + '\n')
        log_file.close()
        # raise
        return '', '',[]

def graph_query(entity_id,iteration=3):
    # print("iteration", iteration)
    try:
        query_str="""PREFIX gas: <http://www.bigdata.com/rdf/gas#>

SELECT ?item ?itemLabel ?linkTo
WHERE
{
  SERVICE gas:service {
    gas:program gas:gasClass "com.bigdata.rdf.graph.analytics.SSSP" ;
                gas:in wd:"""+entity_id+""";
                gas:traversalDirection "Forward" ;
                gas:out ?item ;
                gas:out1 ?depth ;
                gas:maxIterations """+str(iteration)+""" ;
                gas:linkType wdt:P279 .
  }
  OPTIONAL { ?item wdt:P279 ?linkTo }

  SERVICE wikibase:label {bd:serviceParam wikibase:language "en" }
}"""
        sparql.setQuery(query_str)
        sparql.setReturnFormat(JSON)
        # print(sparql.query())
        results = sparql.query().convert()
        # print(results)
        results_df = pd.io.json.json_normalize(results['results']['bindings'])
        # print(results_df[['item.value','linkTo.value']])
        items = list(results_df['item.value'].values) if 'item.value' in results_df else []

        values = list(results_df['itemLabel.value'].values) if 'itemLabel.value' in results_df else []

        linkto_df = list(results_df['linkTo.value'].values) if 'linkTo.value' in results_df else ['']*len(items)
        link_tos=['']*len(items) if len(items)>0 else []
        # print("ENTITY",entity_id)
        link_tos=[linkto_df[i]  if '/' in  str(linkto_df[i]) else '' for i,x in enumerate(link_tos)]
        #if values[i] != x and type(values[i]) == str
        link_tos = [str(x).split('/')[-1] if '/' in str(x) else ''  for i, x in enumerate(link_tos) if values[i] != items[i]] #if values[i] != ids[i]]#if values[i] != ids[i] and type(values[i]) == str
        ids = [str(x).split('/')[-1] for i, x in enumerate(items) if values[i] != x and type(values[i]) == str]
        labels = [x.lower() for i, x in enumerate(values) ]#if x != items[i] and type(x) == str
        # print(entity_id,len(ids),len(link_tos))
        # for i, id in enumerate(ids):
        #     print(i,id, values[i],link_tos[i])
        return ids, labels, link_tos
    except Exception:
        return [],[], []

def get_entities_SPARQ_by_property(entity_id, propertyID="P31",is_forward=True):
    forward_query="""SELECT ?item ?itemLabel 
    WHERE{wd:"""+ entity_id+""" wdt:"""+propertyID + """?item .       
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }}"""
    reverse_query="""SELECT ?item ?itemLabel
    WHERE
    {?item wdt:"""+propertyID+ """ wd:""" + entity_id + """ .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }}"""
    if is_forward:
        query_str=forward_query
    else:
        query_str=reverse_query
    try:
        sparql.setQuery(query_str)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # print(results)
        results_df = pd.io.json.json_normalize(results['results']['bindings'])
        # print(results_df[['item.value']['itemLabel.value']].heading())
        items =list(results_df['item.value'].values) if 'item.value' in results_df else []
        items = [x.split('/')[-1] for x in items]
        values=list(results_df['itemLabel.value'].values) if 'item.value' in results_df else []
        ids = [x.split('/')[-1] for i, x in enumerate(items) if values[i] != x and type(values[i])==str]

        labels = [x.lower() for i, x in enumerate(values) if x != items[i] and type(x)==str]
        #print(ids,labels)
        return ids, labels
    except Exception:
        print("QUERY ERROR",propertyID, entity_id,"forward:",is_forward)
        log_file=open("error_log.txt",'a')
        log_file.write("get_entities_SPARQ_by_property_reverse"+entity_id+';'+propertyID+'\n')
        log_file.close()
        # raise
        return [], []
def remove_empty_linkto(parent_ids,parent_labels,parent_link_tos):
    removed_parent_link_to_idx=[i for i,x in enumerate(parent_link_tos) if x=='']
    parent_ids = [x for i,x in enumerate(parent_ids) if i not in removed_parent_link_to_idx]
    parent_labels = [x for i,x in enumerate(parent_labels) if i not in removed_parent_link_to_idx]
    parent_link_tos=[x for i,x in enumerate(parent_link_tos) if i not in removed_parent_link_to_idx]
    return parent_ids,parent_labels,parent_link_tos

def get_wiki_id_from_text(noun,entity_dict={},iter_num=3):
    # print("iteration",iter_num)
    match_ids, match_labels=text2id(noun)
    for idx, entity_id in enumerate(match_ids):
        if entity_id not in entity_dict:
            entity_label=match_labels[idx]
            parent_ids, parent_labels,parent_link_tos = graph_query(entity_id,iteration=iter_num)#get_entities_SPARQ_by_property_reverse(entity_id,'P279')
            parent_relations=[]
            part_of_ids,part_of_labels = get_entities_SPARQ_by_property(entity_id,  'P361',is_forward=True)
            instance_of_ids, instance_of_labels = get_entities_SPARQ_by_property(entity_id,  'P31',is_forward=True)
            parent_count=len([x for x in parent_link_tos if x!=''])
            if len(instance_of_ids)>0 and parent_count==0:
                parent_ids, parent_labels, parent_link_tos=remove_empty_linkto(parent_ids, parent_labels, parent_link_tos)
                for instance_of_id in instance_of_ids:
                    local_parent_ids, local_parent_labels, local_parent_link_tos = graph_query(instance_of_id, iteration=iter_num)
                    parent_ids.extend(local_parent_ids)
                    parent_labels.extend(local_parent_labels)
                    parent_link_tos.extend(local_parent_link_tos)
                    similar_parent_link_tos = [x for i, x in enumerate(local_parent_link_tos) if
                                               local_parent_ids[i] == instance_of_id]
                    for similar_link_to in similar_parent_link_tos:
                        parent_ids.append(entity_id)
                        parent_labels.append(entity_label)
                        parent_link_tos.append(similar_link_to)
            parent_count = len([x for x in parent_link_tos if x != ''])
            if len(part_of_ids)>0 and parent_count==0:
                parent_ids, parent_labels, parent_link_tos=remove_empty_linkto(parent_ids, parent_labels, parent_link_tos)
                for part_of_id in part_of_ids:
                    local_parent_ids, local_parent_labels, local_parent_link_tos = graph_query(part_of_id, iteration=iter_num)
                    parent_ids.extend(local_parent_ids)
                    parent_labels.extend(local_parent_labels)
                    parent_link_tos.extend(local_parent_link_tos)
                    similar_parent_link_tos = [x for i, x in enumerate(local_parent_link_tos) if
                                               local_parent_ids[i] == part_of_id]
                    for similar_link_to in similar_parent_link_tos:
                        parent_ids.append(entity_id)
                        parent_labels.append(entity_label)
                        parent_link_tos.append(similar_link_to)
            for ii,parent_id in enumerate(parent_ids):
                parent_relations.append({"id":parent_id,"label":parent_labels[ii],"link_to":parent_link_tos[ii]})

            entity_dict[entity_id]={"name_mention":[noun],"label":entity_label,"parents":parent_relations,
                                    "part_ofs":{"ids":part_of_ids,"labels":part_of_labels}
                                    ,"instance_ofs":{"labels":instance_of_labels,"ids":instance_of_ids}}
        else:
            if noun not in entity_dict[entity_id]["name_mention"]:
                entity_dict[entity_id]["name_mention"].append(noun)
    return match_ids

# print(get_wiki_id_from_text("nanometers",{},2))
