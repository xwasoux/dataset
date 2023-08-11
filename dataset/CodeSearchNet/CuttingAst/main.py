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

from astars import ANode, AParser, AstAnalyser, AstOperator, ACodeGenerator
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

def dist2cosSim(distance:int) -> float:
    return 1/(1+distance)

def treeSize(root) -> int:
    return len(root.descendants)+1
    
class OrderList:
    def __init__(self) -> None:
        pass
    
    def front_seq(self, tree:ANode) -> list:
        res = AstAnalyser.forwardSequencialCodeDelete(tree=tree)
        return res

    def back_seq(self, tree:ANode) -> list:
        res = AstAnalyser.backwardSequencialCodeDelete(tree=tree)
        return res

    def rule_point(self, tree:ANode) -> list:
        units = ["if_statement", "elif_clause", "else_clause", 
                 "for_statement", "while_statement", "expression_statement", 
                 "return_statement", "break_statement", "with_statement"]
        res = AstAnalyser.selectedPointingCodeDelete(tree=tree, types=units)
        return res

    def random_point(self, tree:ANode) -> list:
        return 

def createDeleteInfo(baseDict:dict, traversalRes:list, baseDir:str, lang:str, purpose:str, deletionType:str) -> None:
    generator = ACodeGenerator()

    storeJsonl = []
    for target in traversalRes:
        recoverCode = generator.generate(root=target)

        mainDict = deepcopy(baseDict)

        mainDict["editedCode"] = recoverCode
        mainDict["editedSizeChar"] = len(recoverCode)
        mainDict["editedSizeLine"] = len(recoverCode.split("\n"))
        mainDict["editedSizeTree"] = treeSize(target)

        mainDict["levelDistChar"] = Levenshtein.distance(mainDict["originalCode"], recoverCode)
        mainDict["levelDistLine"] = Levenshtein.distance(mainDict["originalCode"].split("\n"), recoverCode.split("\n"))
        mainDict["diffNodeSize"]  = mainDict["originalSizeTree"] - mainDict["editedSizeTree"]

        storeJsonl.append(mainDict)

    corename = baseDict["path"].split("/")[-1].split(".")[0]
    storeFilename = path.join(baseDir, lang, purpose, deletionType, f"{deletionType}_{corename}.jsonl")

    df = pd.DataFrame(storeJsonl)
    df.to_json(storeFilename, orient="records", force_ascii=False, lines=True)

    return None

    
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", type=str)
    parser.add_argument("--all_lang", action="store_true")

    parser.add_argument("--source_base_dir", type=str)
    parser.add_argument("--target_base_dir", type=str)

    parser.add_argument("--pretrained_model", type=str)

    args = parser.parse_args()

    # anyDataTypes = ["front_seq", "back_seq", "rule_point", "random_point"]
    anyDataTypes = ["front_seq", "back_seq", "rule_point"]

    if args.all_lang:
        languages = ["go", "java", "javascript", "php", "python", "ruby"]
    else:
        languages = [args.lang]

    for lang in languages:
        logging.info(f"=== {lang} ===")
        jsonlPaths = getJsonlPaths(path.join(args.source_base_dir, lang))

        mkDir(path.join(args.target_base_dir, lang))
        
        for splitDataPath in jsonlPaths:
            purpose = splitDataPath.split("/")[-1].split(".")[0]
            logging.info(f"== {purpose} ==")
            
            mkDir(path.join(args.target_base_dir, lang, purpose))

            with open(f"{splitDataPath}") as f:
                jsonl = [json.loads(l) for l in f.readlines()]
            
            for line in tqdm(jsonl):
                baseDict = {}

                baseDict["repo"] = line["repo"]
                baseDict["path"] = line["path"]

                code = removeComments(line["original_string"])
                
                baseDict["originalCode"] = code
                baseDict["originalSizeChar"] = len(code)
                baseDict["originalSizeLine"] = len(code.split("\n"))

                parser = AParser()
                tree = parser.parse(text=code, lang=lang)

                baseDict["originalSizeTree"] = treeSize(tree)


                for eachType in anyDataTypes:
                    # logging.info(f"= {eachType} =")
                    orderList = OrderList()
                    getDeletionList = getattr(orderList, eachType)
                    orderTraversalRes = getDeletionList(tree)

                    mkDir(path.join(args.target_base_dir, lang, purpose, eachType))
                    createDeleteInfo(baseDict, orderTraversalRes, args.target_base_dir, lang, purpose, eachType)

    return None

if __name__ == "__main__":
    main()