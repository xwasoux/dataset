#!/usr/bin/env python

import re
import os
import json
import logging
import argparse
import Levenshtein
import pandas as pd
from os import path
from glob import glob
from tqdm import tqdm
from copy import deepcopy

from astars import ANode, AParser
from transformers import AutoTokenizer

from utils import get_jsonl_paths, remove_comments_and_docstrings, remove_spaces_and_tabs, flatten_code, tree_size, distance_to_cosine, get_subtree_elements

logging.basicConfig(format='%(asctime)s -\n %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


    
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", type=str)
    parser.add_argument("--all_lang", action="store_true")
    parser.add_argument("--target_subtree_root", nargs="*")

    parser.add_argument("--source_base_dir", type=str)
    parser.add_argument("--target_base_dir", type=str)

    parser.add_argument("--pretrained_model", type=str)

    args = parser.parse_args()

    if args.all_lang:
        languages = ["go", "java", "javascript", "php", "python", "ruby"]
    else:
        languages = [args.lang]

    for lang in languages:
        logging.info(f"=== {lang} ===")
        jsonl_paths = get_jsonl_paths(path.join(args.source_base_dir, lang))

        os.makedirs(path.join(args.target_base_dir, lang), exist_ok=True)
        
        for each_partition_path in jsonl_paths:
            partition = each_partition_path.split("/")[-1].split(".")[0]
            logging.info(f"== {partition} ==")
            
            with open(f"{each_partition_path}") as f:
                jsonl = [json.loads(l) for l in f.readlines()]
            
            for line in tqdm(jsonl):
                tree = AParser.parse(text=line["cleaned_code"], lang=lang)

                cleaned_code_subtree_elements = get_subtree_elements(tree, args.target_subtree_root)
                line["cleaned_code_subtree_elements"] = cleaned_code_subtree_elements
                line["cleaned_code_subtree_elements_unique"] = list(set(cleaned_code_subtree_elements))

            df = pd.DataFrame(jsonl)
            stored_filename = path.join(args.target_base_dir, lang, f"{partition}.jsonl")
            df.to_json(stored_filename, orient="records", force_ascii=False, lines=True)

    return None

if __name__ == "__main__":
    main()
