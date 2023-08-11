lang=python
source_base_dir=../data/workbench/Extracted
target_base_dir=../output/SubtreeElement
pretrained_model=microsoft/codebert-base

python3 ../main.py \
    --lang $lang \
    --source_base_dir $source_base_dir \
    --target_base_dir $target_base_dir \
    --pretrained_model $pretrained_model

