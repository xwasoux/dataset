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

from astars import ANode, AParser, AstAnalyser, AstOperator, ACodeGenerator, AllNodeTraverser, ANamedTraverser, ATypeTraverser
from transformers import AutoTokenizer

logging.basicConfig(format='%(asctime)s -\n %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def get_jsonl_paths(pathName:str) -> list:
    condition = f'{pathName}/*.jsonl'
    return glob(condition, recursive=True)


def remove_comments(code) -> str:
    code = re.sub(r'\"\"\"(.|\n)*?\"\"\"', '', code)   # """comments"""
    code = re.sub(r"\'\'\'(.|\n)*?\'\'\'", '', code)   # '''comments'''
    code = re.sub(r'\#.*', '', code)                   ##comments
    return code

def get_subtree_elements(code:str, lang:str) -> list:
    units = ["if_statement", "elif_clause", "else_clause", 
                "for_statement", "while_statement", "expression_statement", 
                "return_statement", "break_statement", "with_statement"]

    parser = AParser()
    tree = parser.parse(code, lang)

    traverse_res = ATypeTraverser.leftPreOrder(tree)
    subtree_elem = [node for node in traverse_res if node in units]

    return subtree_elem

    
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", type=str)
    parser.add_argument("--all_lang", action="store_true")

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
        
        for each_partition in jsonl_paths:
            partition = each_partition.split("/")[-1].split(".")[0]
            logging.info(f"== {partition} ==")
            
            with open(f"{each_partition}") as f:
                jsonl = [json.loads(l) for l in f.readlines()]
            
            for line in tqdm(jsonl):
                code = remove_comments(line["original_string"])
                line["code_pure"] = code
                line["subtree_elements"] = get_subtree_elements(code, lang)
                line["subtree_elements_unique"] = list(dict.fromkeys(line["subtree_elements"]))

            df = pd.DataFrame(jsonl)
            stored_filename = path.join(args.target_base_dir, lang, f"{partition}.jsonl")
            df.to_json(stored_filename, orient="records", force_ascii=False, lines=True)

    return None

if __name__ == "__main__":
    main()
