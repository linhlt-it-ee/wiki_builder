# How to use

Input: `reuters.csv` consists of >10000 news labelled with their topics. Each news can belong to several topics.

**1. Crawl relating wiki entities from name mentions in each news.**

Input: `reuters.csv` and its directory (e.g. `data`).

Output:

```
data
├── reuters.csv
├── reuters_all_entity.json
├── reuters_all_entity.pck
├── reuters_all_entity_brief.json
├── reuters_all_entity_level.json
├── reuters_all_entity_parent_leaf.pck
├── reuters_all_entity_patent_entity_relations.json
└── reuters_entities
    ├── name_mention_1.txt
    ├── name_mention_1.txt_entity_dict_iteration.pck
    ├── name_mention_1.txt_entity_dict_not_found_iteration.pck
    └── ...
```

Please refer to `./crawl_wiki_from_reuters.ipynb`.

**2. Create relations between a news and wiki entities as edges:**
- Edge weight is 0, if the entities are the name mentions in news.
- Edge weight is 1, if the entites are the parent 1 of a name mention in news.
- ...
- Edge weight is `d`, if the entites are the parent `d` of a name mention in news.

Input: folder `reuters_entities` and file `reuters_all_entity.json`

Output: 

```
data
├── doc_vs_concepts.json
├── reuters_all_entity.json
```

Please refer to `../extract_edge.py` or use `python3 ../extract_edge.py` to re-produce the results.

**3. Create graph for node classification: each node is a news, and node label is its topics.**

Input: `reuters_all_entity_brief.json` and `doc_vs_concepts.json`

Please refer to `../main.py` or use `python3 main.py` to re-produce the results.
