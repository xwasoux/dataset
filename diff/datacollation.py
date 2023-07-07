import glob
import git
import Levenshtein
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_link", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()


if __name__ == "__main__":
    main()
