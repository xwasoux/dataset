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

from astars import AParser, AstAnalyser, AstOperator, ACodeGenerator
from transformers import AutoTokenizer

logging.basicConfig(format='%(asctime)s -\n %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def get_jsonl_path_list(base_dir:str) -> list:
    condition = f'{base_dir}/*.jsonl'
    return glob(condition, recursive=True)

def remove_comments(code) -> str:
    code = re.sub(r'\"\"\"(.|\n)*?\"\"\"', '', code)   # """comments"""
    code = re.sub(r"\'\'\'(.|\n)*?\'\'\'", '', code)   # '''comments'''
    code = re.sub(r'\#.*', '', code)                   ##comments
    return code

def mk_dir(path_str:str) -> None:
    try:
        os.mkdir(path_str)
    except:
        pass
    return None
    

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

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    for lang in languages:
        logging.info(f"=== {lang} ===")
        jsonl_path_list = get_jsonl_path_list(path.join(args.source_base_dir, lang))
        logging.info(jsonl_path_list)

        mk_dir(path.join(args.target_base_dir, lang))
        
        for jsonl_path in jsonl_path_list:
            purpose = jsonl_path.split("/")[-1].split(".")[0]
            logging.info(f"Purpose : {purpose}")
            

            with open(f"{jsonl_path}") as f:
                jsonl = [json.loads(l) for l in f.readlines()]
            
            extracted_jsonl = []
            for line in tqdm(jsonl):
                code = remove_comments(line["original_string"])
                code_tokens = tokenizer.tokenize(code)

                if len(code_tokens) <= 510:
                    extracted_jsonl.append(line)
            logging.info(f"New jsonl : {len(extracted_jsonl)}")

            stored_file_path = path.join(args.target_base_dir, lang, f"{purpose}.jsonl")
            df = pd.DataFrame(extracted_jsonl)
            df.to_json(stored_file_path, force_ascii=False, lines=True, orient="records")

    return None


if __name__ == "__main__":
    main()