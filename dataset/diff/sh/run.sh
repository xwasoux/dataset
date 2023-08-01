github_link=##repository

cd ..
mkdir diff-commit workbench
python3 datacollation.py \
    --repo_link $github_link \
    --output_dir diff/##name

cd sh/