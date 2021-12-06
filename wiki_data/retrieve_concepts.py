import json
import logging
import math
import os
import threading
from typing import Dict, List

import data_crawl.utils as data_utils
from tqdm import tqdm

import utils
from args import *


def retrieve_concepts(doc_path: str, cache_dir: str, lang: str):
    os.makedirs(cache_dir, exist_ok=True)
    doc_mention_path = os.path.join(cache_dir, "doc_mention.json")
    mention_dir = os.path.join(cache_dir, "mentions")

    # Extract name mentions
    doc_mention = extract_name_mentions(doc_path, lang=lang)
    utils.dump_json(doc_mention, doc_mention_path)
    # Split them to multiple files
    num_files = 50
    mention = list(set(e for x in doc_mention.values() for e in x))
    split_name_mention_list(mention, mention_dir, num_files)
    # Retrieve concepts of name mentions
    search_wiki_with_threads(mention_dir, 0, num_files, iteration=3)

    return None


def extract_name_mentions(data_path: str, lang: str = "en") -> Dict[str, List[str]]:
    with open(data_path, "r") as f:
        docs = [json.loads(line) for line in f]
    name_mention = {}
    for doc in tqdm(docs, desc="Extract nouns"):
        if lang == "en":
            nouns = data_utils.get_nouns_nltk(doc["1st_claim"], ngram_range=3)
        elif lang == "ja":
            nouns = data_utils.get_nouns_janome(doc["content"])
        else:
            raise NotImplementedError
        name_mention[doc["id"]] = [x.lower() for x in nouns]
    return name_mention


def split_name_mention_list(mention_list, mention_dir, num_files):
    total = len(mention_list)
    line_num_each_file = int(math.ceil(total / num_files))
    os.makedirs(mention_dir, exist_ok=True)

    for i in range(num_files):
        fname = os.path.join(mention_dir, str(i) + ".txt")
        with open(fname, "w") as f:
            if i * line_num_each_file + line_num_each_file < total - 1:
                start = i * line_num_each_file
                end = i * line_num_each_file + line_num_each_file
                logging.info(f"File {fname}: start {start} - end {end}")
                for j in mention_list[start:end]:
                    f.write(j + "\n")
            else:
                for j in mention_list[i * line_num_each_file :]:
                    f.write(j + "\n")


def search_wiki_with_threads(mention_dir, start, end, iteration=2):
    mention_paths = utils.get_file_name_in_dir(mention_dir, "txt")
    threads = []
    for i, file_name in enumerate(mention_paths[start:end]):
        base_name = os.path.splitext(file_name)[0]
        entities_file = base_name + "_entities.pck"
        not_found_entities_file = base_name + "_entities_not_found.pck"
        thread1 = threading.Thread(
            target=search_wiki_with_forward_iteration,
            args=(file_name, entities_file, not_found_entities_file, iteration),
        )
        thread1.start()
        threads.append(thread1)
        logging.info(f"THREAD {i} START")
    for thread in threads:
        thread.join()


def search_wiki_with_forward_iteration(
    mention_path, output_entity_file, not_wiki_output, lang="en", iter_num=3
):
    mentions = open(mention_path, "r").readlines()
    not_found_entity = []
    total = len(mentions)
    entity_dict = {}
    for i, mention in enumerate(mentions):
        word = data_utils.remove_characters("\n", "", mention)
        word = data_utils.remove_characters("-", " ", word)
        entities = data_utils.get_wiki_id_from_text(word, entity_dict, iter_num)
        if lang == "en":
            singu_word = data_utils.convert_plural_to_singular(word)
            if singu_word != word:
                entities.extend(data_utils.get_wiki_id_from_text(singu_word, entity_dict, iter_num))
        if len(entities) == 0:
            not_found_entity.append(mention)
        # temporary results
        utils.dump(entity_dict, output_entity_file)
        utils.dump(not_found_entity, not_wiki_output)
        logging.debug(f"Finished {i}/{total}")

    utils.dump(entity_dict, output_entity_file)
    utils.dump(not_found_entity, not_wiki_output)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(module)s %(message)s", level=logging.DEBUG)
    args = make_data_args()
    retrieve_concepts(args.data_path, args.cache_dir, lang=args.lang)
