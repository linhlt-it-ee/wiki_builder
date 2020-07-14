# wiki_builder
A - Knowledge Graph Crawling from Wiki 
\
Step 0: Find name mention in description or short name of an entity in the wikidata
Ex: trifluoromethyl Q2302144

\
Method 1:
1.	Step 1: Find direct parent of each entity for upper level 1
2.	Step 2: Find parent intersection from all parents of each entity to find a root node
"""SELECT ?item ?itemLabel WHERE { wd:"""+entity_id1+""" wdt:P279+ ?item . ?item  wdt:P279+ wd:"""+entity_id2+""" . SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }}"""
2.1.	 Find a path between 2 entities. For example: there is a path between entity1:semiconductor device and entity2: electric device: semiconductor device>>electronic component>>chemical element>>electronic device (it could takes time or make time out request when an entity have too many parents)
2.2.	 Check whether entity_id1 is direct parent of entity_id2
"""SELECT ?item ?itemLabel WHERE{wd:"""+ entity_id+""" wdt:"""+propertyID + """?item . 
SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }}"""
     Disadvantage: Timeout request error or malfunction request when we have a few threads to crawl data.
   This method could not be implemented any more because or timeout request or too long to wait (at least 4 days) 
 
 \
Method 2:
1.	Step 1: Find parent in upper 3 level and the link between all nodes in only 1 step instead of find direct parent for 3 times which leading to timeout request or malfunction request
""PREFIX gas: <http://www.bigdata.com/rdf/gas#>

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
  

2.	Update label, short_name, description for all upper level parents
""" SELECT
  ?id?label?desc
  (GROUP_CONCAT(DISTINCT(?aka); separator="|") AS ?akas)
WHERE{
  VALUES ?id { wd:"""+entity_id+""" }
  OPTIONAL{ ?id rdfs:label ?label. FILTER(LANG(?label)="en")}
  OPTIONAL{ ?id skos:altLabel ?aka . FILTER(LANG(?aka) = "en")}
  OPTIONAL{ ?id schema:description ?desc . FILTER(LANG(?desc) = "en")}
}
GROUP BY ?id?label?desc"""
3.	Make tree builder from all links between entities with node level 0 = root 
Advantage: Fast and prevent “timeout request”
B – Knowledge Graph Level Labelling
Problem 1: How to decide level of an entity
 
 
Current solution: Choose a lower level = 5 if entity_1 is a child of both entity_2 (level3) and entity_3 (level4)


