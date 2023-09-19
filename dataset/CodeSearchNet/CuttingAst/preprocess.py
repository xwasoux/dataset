#!/usr/bin/env python

import os
import json
import logging
import argparse
import pandas as pd
from os import path
from glob import glob
from tqdm import tqdm

from transformers import AutoTokenizer

from utils import remove_comments_and_docstrings, remove_spaces_and_tabs, flatten_code

logging.basicConfig(format='%(asctime)s -\n %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def get_jsonl_path_list(base_dir:str) -> list:
    condition = f'{base_dir}/*.jsonl'
    return glob(condition, recursive=True)


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

        os.makedirs(path.join(args.target_base_dir, lang), exist_ok=True)
        
        for jsonl_path in jsonl_path_list:
            partition = jsonl_path.split("/")[-1].split(".")[0]
            logging.info(f"Partition : {partition}")
            
            with open(f"{jsonl_path}") as f:
                jsonl = [json.loads(l) for l in f.readlines()]
            
            extracted_jsonl = []
            for line in tqdm(jsonl):
                cleaned_code = remove_comments_and_docstrings(source=line["original_string"], lang=lang)
                line["cleaned_code"] = cleaned_code
                code_subtokens = tokenizer.tokenize(cleaned_code)
                line["code_subtokens"] = code_subtokens

                code_noindent = remove_spaces_and_tabs(cleaned_code)
                line["code_noindent"] = code_noindent
                line["code_noindent_subtokens"] = tokenizer.tokenize(code_noindent)

                flattened_code = flatten_code(code_noindent)
                line["flattened_code"] = flattened_code
                line["flattened_code_subtokens"] = tokenizer.tokenize(flattened_code)

                if len(code_subtokens) <= 510:
                    extracted_jsonl.append(line)

            logging.info(f"New jsonl : {len(extracted_jsonl)}")

            df = pd.DataFrame(extracted_jsonl)

            stored_file_path = path.join(args.target_base_dir, lang, f"{partition}")
            df.to_json(f"{stored_file_path}.jsonl", force_ascii=False, lines=True, orient="records")
            df.to_csv(f"{stored_file_path}.csv")

    return None


if __name__ == "__main__":
    main()