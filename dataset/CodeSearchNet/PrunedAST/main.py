#!/usr/bin/env python

import os
import json
import logging
import argparse
import Levenshtein
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from .utils import get_jsonl_paths, remove_spaces_and_tabs, flatten_code, tree_size, distance_to_cosine

from astars import AParser,AParseTree, ATraverser, APruner

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

class Pruner:
    def __init__(self) -> None:
        pass
    
    def sequence_forward(self, args:argparse, tree:AParseTree, base_dict:dict) -> dict:
        pruned_res = APruner.seqForwardPrune(tree=tree)
        return append_ast_cut_dict(base_dict=base_dict, pruned_res=pruned_res)

    def sequence_backward(self, args:argparse, tree:AParseTree, base_dict:dict) -> dict:
        pruned_res = APruner.seqBackwardPrune(tree=tree)
        return append_ast_cut_dict(base_dict=base_dict, pruned_res=pruned_res)

    def point_rule(self, args:argparse, tree:AParseTree, base_dict:dict) -> dict:
        pruned_res = APruner.selectedPointingPrune(tree=tree, selections=args.target_subtree_root)
        return append_ast_cut_dict(base_dict=base_dict, pruned_res=pruned_res)

    def point_all(self, args:argparse, tree:AParseTree, base_dict:dict) -> dict:
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
        main_dict["edited_code_render"] = str(recover_code)
        main_dict["edited_code_char_size"] = len(recover_code)
        main_dict["edited_code_line_size"] = len(recover_code.split("\n"))
        main_dict["edited_code_tree_size"] = tree_size(pruned_ast)

        traverser = ATraverser()

        pruned_ast_res = traverser.preorderTraverse(pruned_ast)
        main_dict["edited_ast_node_types"] = list(set(pruned_ast_res.preNodeTypes))

        main_dict["pruned_node_type"] = subtree.type
        main_dict["pruned_node_is_named"] = subtree.is_named

        main_dict["cleaned_code_diff_char_size"] = Levenshtein.distance(main_dict["cleaned_code"], recover_code)
        main_dict["cleaned_code_diff_line_size"] = Levenshtein.distance(main_dict["cleaned_code"].split("\n"), recover_code.split("\n"))
        main_dict["cleaned_code_diff_size_node"]  = main_dict["cleaned_code_tree_size"] - main_dict["edited_code_tree_size"]

        main_dict["cleaned_code_cosine_char"] = distance_to_cosine(main_dict["cleaned_code_diff_char_size"])
        main_dict["cleaned_code_cosine_line"] = distance_to_cosine(main_dict["cleaned_code_diff_line_size"])
        main_dict["cleaned_code_cosine_node"] = distance_to_cosine(main_dict["cleaned_code_diff_size_node"])
        
    return stored_jsonl


    
def main() -> None:
    '''This code create Cutting-AST dataset.

    Args: 
        None
        
    Returns: 
        None
        
    Note: 
        Here is a Dataset structure.
        
        {
            "repo":                         repository name, 
            "path":                         file path name in the repository, 
            "func_name":                    function name in the repository, 
            "original_string":              original function, 
            "cleaned_code":                 removed comments and docstrings, 
            "cleaned_code_char_size":       number of character, 
            "cleaned_code_line_size":       number of line, 
            "cleaned_code_tree_size":       size of tree, 
            "cleaned_code_ast_node_types":  a list of node types, 
            "edited_code":                  an edited code using node pruning, 
            "edited_code_char_size":        number of character, 
            "edited_code_line_size":        number of line, 
            "edited_code_tree_size":        size of tree, 
            "edited_ast_node_types":        a list of node types, 
            "diff_char_size":               levenshtein distance of character between cleaned_code & edited_code, 
            "diff_line_size":               levenshtein distance of line between cleaned_code & edited_code, 
            "diff_size_node":               difference of tree size between cleaned_code & edited_code, 
            "cosine_char":            conine similarity calculated using levenshtein distance of character, 
            "cosine_line":            conine similarity calculated using levenshtein distance of line, 
            "cosine_node":            conine similarity calculated using dff of tree size
        }

    '''
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", type=str)
    parser.add_argument("--all_lang", action="store_true")
    parser.add_argument("--target_subtree_root", nargs="*")

    parser.add_argument("--source_base_dir", type=str)
    parser.add_argument("--target_base_dir", type=str)

    parser.add_argument("--pretrained_model", type=str)

    parser.add_argument("--upper_data_size", type=int)

    args = parser.parse_args()

    pruning_types = ("sequence_forward", "sequence_backward", "point_rule", "point_all")

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

                for pruning in pruning_types:
                    # logging.info(f"= {pruning} =")
                    pruner = Pruner()
                    get_pruned_res = getattr(pruner, pruning)
                    stored_jsonl = get_pruned_res(args, tree, line)

                    os.makedirs(os.path.join(args.target_base_dir, lang, partition, pruning), exist_ok=True)

                    corename = line["path"].split("/")[-1].split(".")[0]
                    stored_filename = os.path.join(args.target_base_dir, lang, partition, pruning, f"{pruning}_{corename}")

                    df = pd.DataFrame(stored_jsonl)
                    df.to_json(f"{stored_filename}.jsonl", orient="records", force_ascii=False, lines=True)
                    df.to_csv(f"{stored_filename}.csv")
                
    return None

if __name__ == "__main__":
    main()