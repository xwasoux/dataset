import glob
import git
import logging
import Levenshtein
import argparse

logging.basicConfig(format='%(asctime)s -\n %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_link", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()


if __name__ == "__main__":
    main()
