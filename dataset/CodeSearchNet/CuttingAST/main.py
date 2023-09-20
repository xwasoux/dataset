#!/usr/bin/env python

import os
import json
import logging
import argparse
import Levenshtein
import pandas as pd
from glob import glob
from tqdm import tqdm
from copy import deepcopy

from astars import AParser,AParseTree, ATraverser, APruner

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def get_jsonl_paths(base_dir:str) -> list:
    condition = f'{base_dir}/*.jsonl'
    return glob(condition, recursive=True)

def tree_size(tree:AParseTree) -> int:
    return len(tree.root.descendants)+1
    
def create_base_dict(original_dict:dict, tree:AParseTree) -> dict:
    base_dict = {}

    base_dict["repo"] = original_dict["repo"]
    base_dict["path"] = original_dict["path"]
    base_dict["func_name"] = original_dict["func_name"]
    base_dict["original_string"] = original_dict["original_string"]

    cleaned_code = original_dict["cleaned_code"]
    base_dict["cleaned_code"] = cleaned_code
    base_dict["cleaned_code_size_char"] = len(cleaned_code)
    base_dict["cleaned_code_size_line"] = len(cleaned_code.split("\n"))
    base_dict["cleaned_code_size_tree"] = tree_size(tree)

    traverser = ATraverser()
    res = traverser.preorderTraverse(tree)
    base_dict["cleaned_code_ast_node_types"] = list(set(res.preNodeTypes))
                
    return base_dict

def dist2cosSim(distance:int) -> float:
    return 1/(1+distance)

class Pruner:
    def __init__(self) -> None:
        pass
    
    def front_seq(self, args:argparse, tree:AParseTree, base_dict:dict) -> dict:
        pruned_res = APruner.seqForwardPrune(tree=tree)
        return append_ast_cut_dict(base_dict=base_dict, pruned_res=pruned_res)

    def back_seq(self, args:argparse, tree:AParseTree, base_dict:dict) -> dict:
        pruned_res = APruner.seqBackwardPrune(tree=tree)
        return append_ast_cut_dict(base_dict=base_dict, pruned_res=pruned_res)

    def rule_point(self, args:argparse, tree:AParseTree, base_dict:dict) -> dict:
        pruned_res = APruner.selectedPointingPrune(tree=tree, selections=args.target_subtree_root)
        return append_ast_cut_dict(base_dict=base_dict, pruned_res=pruned_res)

    def all_point(self, args:argparse, tree:AParseTree, base_dict:dict) -> dict:
        pruned_res = APruner.seqPointingPrune(tree=tree)
        return append_ast_cut_dict(base_dict=base_dict, pruned_res=pruned_res)

def append_ast_cut_dict(base_dict:dict, pruned_res:tuple) -> list:
    stored_jsonl = []

    for res_pair in pruned_res:
        pruned_ast = res_pair[0]
        subtree = res_pair[1]

        main_dict = deepcopy(base_dict)
        recover_code = pruned_ast.recover()

        main_dict["edited_code"] = recover_code
        main_dict["edited_code_size_char"] = len(recover_code)
        main_dict["edited_code_size_line"] = len(recover_code.split("\n"))
        main_dict["edited_code_size_tree"] = tree_size(pruned_ast)

        main_dict["diff_size_char"] = Levenshtein.distance(main_dict["cleaned_code"], recover_code)
        main_dict["diff_size_line"] = Levenshtein.distance(main_dict["cleaned_code"].split("\n"), recover_code.split("\n"))
        main_dict["diff_size_node"]  = main_dict["cleaned_code_size_tree"] - main_dict["edited_code_size_tree"]
        
        traverser = ATraverser()
        res = traverser.preorderTraverse(pruned_ast)
        main_dict["edited_ast_node_types"] = list(set(res.preNodeTypes))

        stored_jsonl.append(main_dict)

    return stored_jsonl


    
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", type=str)
    parser.add_argument("--all_lang", action="store_true")
    parser.add_argument("--target_subtree_root", nargs="*")

    parser.add_argument("--source_base_dir", type=str)
    parser.add_argument("--target_base_dir", type=str)

    parser.add_argument("--pretrained_model", type=str)

    parser.add_argument("--upper_data_size", type=int)

    args = parser.parse_args()

    deletion_types = ["front_seq", "back_seq", "rule_point", "all_point"]

    if args.all_lang:
        languages = ["go", "java", "javascript", "php", "python", "ruby"]
    else:
        languages = [args.lang]
    
    if args.upper_data_size:
        UPPER_LIMIT = args.upper_data_size
    else:
        UPPER_LIMIT = 100

    for lang in languages:
        logging.info(f"=== {lang} ===")
        retrive_jsonl_paths = get_jsonl_paths(os.path.join(args.source_base_dir, lang))

        os.makedirs(os.path.join(args.target_base_dir, lang), exist_ok=True)
        
        for jsonl_path in retrive_jsonl_paths:
            partition = jsonl_path.split("/")[-1].split(".")[0]
            logging.info(f"== {partition} ==")
            
            os.makedirs(os.path.join(args.target_base_dir, lang, partition), exist_ok=True)

            with open(f"{jsonl_path}") as f:
                jsonl = [json.loads(l) for l in f.readlines()]
            
            for num, _ in enumerate(tqdm(range(UPPER_LIMIT))):
                line = jsonl[num]

                tree = AParser.parse(text=line["cleaned_code"], lang=lang)
                base_dict = create_base_dict(original_dict=line, tree=tree)

                for each_type in deletion_types:
                    # logging.info(f"= {each_type} =")
                    pruner = Pruner()
                    get_pruned_res = getattr(pruner, each_type)
                    stored_jsonl = get_pruned_res(args, tree, base_dict)

                    os.makedirs(os.path.join(args.target_base_dir, lang, partition, each_type), exist_ok=True)

                    corename = base_dict["path"].split("/")[-1].split(".")[0]
                    stored_filename = os.path.join(args.target_base_dir, lang, partition, each_type, f"{each_type}_{corename}")

                    df = pd.DataFrame(stored_jsonl)
                    df.to_json(f"{stored_filename}.jsonl", orient="records", force_ascii=False, lines=True)
                    df.to_csv(f"{stored_filename}.csv")
                
    return None

if __name__ == "__main__":
    main()