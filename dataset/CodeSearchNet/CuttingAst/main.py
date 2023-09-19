#!/usr/bin/env python

import re
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
from transformers import AutoTokenizer

logging.basicConfig(format='%(asctime)s -\n %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def get_jsonl_paths(base_dir:str) -> list:
    condition = f'{base_dir}/*.jsonl'
    return glob(condition, recursive=True)


def remove_comments(code) -> str:
    code = re.sub(r'\"\"\"(.|\n)*?\"\"\"', '', code)   # """comments"""
    code = re.sub(r"\'\'\'(.|\n)*?\'\'\'", '', code)   # '''comments'''
    code = re.sub(r'\#.*', '', code)                   ##comments
    return code

def dist2cosSim(distance:int) -> float:
    return 1/(1+distance)

def tree_size(root) -> int:
    return len(root.descendants)+1
    
class OrderList:
    def __init__(self) -> None:
        pass
    
    def front_seq(self, args:argparse, tree:AParseTree) -> "APruner":
        res = APruner.seqForwardPrune(tree=tree)
        return res

    def back_seq(self, args:argparse, tree:AParseTree) -> "APruner":
        res = APruner.seqBackwardPrune(tree=tree)
        return res

    def rule_point(self, args:argparse, tree:AParseTree, selections:list) -> "APruner":
        res = APruner.selectedPointingPrune(tree=tree, selections=selections)
        return res

    def all_point(self, args:argparse, tree:AParseTree) -> "APruner":
        res = APruner.seqPointingPrune(tree=tree)
        return res

def create_delete_dict(base_dict:dict, traverse_res:list, base_path:str, lang:str, partition:str, deletion_type:str) -> None:
    generator = ACodeGenerator()

    stored_jsonl = []
    for target in traverse_res:
        recover_code = generator.generate(root=target)

        main_dict = deepcopy(base_dict)

        main_dict["edited_code"] = recover_code
        main_dict["edited_size_char"] = len(recover_code)
        main_dict["edited_size_line"] = len(recover_code.split("\n"))
        main_dict["edited_size_tree"] = tree_size(target)

        main_dict["leven_distance_char"] = Levenshtein.distance(main_dict["original_code"], recover_code)
        main_dict["leven_distance_line"] = Levenshtein.distance(main_dict["original_code"].split("\n"), recover_code.split("\n"))
        main_dict["diff_node_size"]  = main_dict["original_size_tree"] - main_dict["edited_size_tree"]

        stored_jsonl.append(main_dict)

    corename = base_dict["path"].split("/")[-1].split(".")[0]
    stored_filename = os.path.join(base_path, lang, partition, deletion_type, f"{deletion_type}_{corename}.jsonl")

    df = pd.DataFrame(stored_jsonl)
    df.to_json(stored_filename, orient="records", force_ascii=False, lines=True)

    return None

def point_delete_dict(args:argparse, base_dict:dict, edit_res:list, base_path:str, lang:str, partition:str, deletion_type:str) -> None:
    generator = ACodeGenerator()

    stored_jsonl = []
    for target in tqdm(edit_res):
        edited_tree = target[0]
        target_node = target[1]
        target_node_name = target_node.type
        recover_code = generator.generate(root=edited_tree)

        main_dict = deepcopy(base_dict)

        main_dict["edited_code"] = recover_code
        main_dict["edited_size_char"] = len(recover_code)
        main_dict["edited_size_line"] = len(recover_code.split("\n"))
        main_dict["edited_size_tree"] = tree_size(edited_tree)

        main_dict["leven_distance_char"] = Levenshtein.distance(main_dict["original_code"], recover_code)
        main_dict["leven_distance_line"] = Levenshtein.distance(main_dict["original_code"].split("\n"), recover_code.split("\n"))
        main_dict["diff_node_size"]  = main_dict["original_size_tree"] - main_dict["edited_size_tree"]

        main_dict["delete_target_node"] = target_node_name

        all_node_ids = AIDTraverser.leftPostOrder(edited_tree)
        all_node_name = []
        for node_id in all_node_ids:
            dup_tree = deepcopy(edited_tree)
            target_node = ASearcher.searchNode(dup_tree, str(node_id))
            all_node_name.append(target_node.type)

        main_dict["edited_ast_node_types"] = list(set(all_node_name))

        stored_jsonl.append(main_dict)

    corename = base_dict["path"].split("/")[-1].split(".")[0]
    stored_filename = os.path.join(base_path, lang, partition, deletion_type, f"{deletion_type}_{corename}.jsonl")

    df = pd.DataFrame(stored_jsonl)
    df.to_json(stored_filename, orient="records", force_ascii=False, lines=True)

    return None

    
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", type=str)
    parser.add_argument("--all_lang", action="store_true")
    parser.add_argument("--target_subtree_root", nargs="*")

    parser.add_argument("--source_base_dir", type=str)
    parser.add_argument("--target_base_dir", type=str)

    parser.add_argument("--pretrained_model", type=str)

    args = parser.parse_args()

    deletion_types = ["front_seq", "back_seq", "rule_point", "all_point"]

    if args.all_lang:
        languages = ["go", "java", "javascript", "php", "python", "ruby"]
    else:
        languages = [args.lang]

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
            
            UPPER_LIMIT = 100
            for num, _ in enumerate(tqdm(range(UPPER_LIMIT))):
                line = jsonl[num]
                base_dict = {}

                base_dict["repo"] = line["repo"]
                base_dict["path"] = line["path"]

                code = remove_comments(line["original_string"])
                
                base_dict["original_code"] = code
                base_dict["original_size_char"] = len(code)
                base_dict["original_size_line"] = len(code.split("\n"))

                parser = AParser()
                tree = parser.parse(text=code, lang=lang)

                base_dict["original_size_tree"] = tree_size(tree)

                ################################
                ## for all node point deletion 
                ################################
                all_node_ids = AIDTraverser.leftPostOrder(tree)

                edit_res = []
                all_node_name = []
                for node_id in all_node_ids:
                    dup_tree = deepcopy(tree)
                    target_node = ASearcher.searchNode(dup_tree, str(node_id))
                    all_node_name.append(target_node.type)

                    if not target_node.is_root:
                        edited_tree = AstOperator.delete(root=dup_tree, target=target_node)
                        edit_res.append((edited_tree, target_node))
                
                base_dict["original_ast_node_types"] = list(set(all_node_name))
                
                os.makedirs(os.path.join(args.target_base_dir, lang, partition, "all_point"), exist_ok=True)
                point_delete_dict(args, base_dict, edit_res, args.target_base_dir, lang, partition, "all_point")


    return None

if __name__ == "__main__":
    main()