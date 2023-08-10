#!/usr/bin/env python

import re
import os
import json
import logging
import argparse
import pandas as pd
from os import path
from glob import glob
from tqdm import tqdm
from copy import deepcopy

from transformers import AutoTokenizer

logging.basicConfig(format='%(asctime)s -\n %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def getJsonlPaths(pathName:str) -> list:
    condition = f'{pathName}/*.jsonl'
    return glob(condition, recursive=True)

def removeComments(code) -> str:
    code = re.sub(r'\"\"\"(.|\n)*?\"\"\"', '', code)   # """comments"""
    code = re.sub(r"\'\'\'(.|\n)*?\'\'\'", '', code)   # '''comments'''
    code = re.sub(r'\#.*', '', code)                   ##comments
    return code

def mkDir(pathStr:str) -> None:
    try:
        os.mkdir(pathStr)
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
        jsonlPaths = getJsonlPaths(path.join(args.source_base_dir, lang))
        logging.info(jsonlPaths)

        mkDir(path.join(args.target_base_dir, lang))
        
        for jsonlFile in jsonlPaths:
            purpose = jsonlFile.split("/")[-1].split(".")[0]
            logging.info(f"Purpose : {purpose}")
            

            with open(f"{jsonlFile}") as f:
                jsonl = [json.loads(l) for l in f.readlines()]
            
            extractedJonl = []
            for line in tqdm(jsonl):
                code = removeComments(line["original_string"])
                code_tokens = tokenizer.tokenize(code)

                if len(code_tokens) <= 510:
                    extractedJonl.append(line)
            logging.info(f"New jsonl : {len(extractedJonl)}")

            storeFile = path.join(args.target_base_dir, lang, f"{purpose}.jsonl")
            df = pd.DataFrame(extractedJonl)
            df.to_json(storeFile, force_ascii=False, lines=True, orient="records")

    return None


if __name__ == "__main__":
    main()
