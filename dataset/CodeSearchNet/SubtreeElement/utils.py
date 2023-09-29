import re
from io import StringIO
import  tokenize
from glob import glob

from astars import ANode, AParseTree, ATraverser

def get_jsonl_paths(base_dir:str) -> list:
    condition = f'{base_dir}/*.jsonl'
    return glob(condition, recursive=True)

def remove_comments_and_docstrings(source:str, lang:str) -> str:
    if lang in ["python"]:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def remove_spaces_and_tabs(source:str) -> str:
    source = re.sub(r" +", " ", source)
    source = re.sub(r"\t+", "\t", source)
    return source

def flatten_code(source:str) -> str:
    source = re.sub(r"\n", " ", source)
    return source

def tree_size(tree:AParseTree) -> int:
    return len(tree.root.descendants)+1
    
def distance_to_cosine(distance:int) -> float:
    return 1/(1+distance)

def get_subtree_elements(tree:AParseTree, target_units:list) -> list:

    traverser = ATraverser()
    res = traverser.preorderTraverse(tree)
    subtree_elem = [node for node in res.preNodeTypes if node in target_units]

    return subtree_elem