import logging
import argparse
import astars
from astars import AParser, AstOperator, ACodeGenerator
from glob import glob

logging.basicConfig(format='%(asctime)s -\n %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str)
    parser.add_argument("--basedir", type=str)

    args = parser.parse_args()

    

    return None

if __name__ == "__main__":
    main()