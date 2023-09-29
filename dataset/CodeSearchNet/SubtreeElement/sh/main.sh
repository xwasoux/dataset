lang=python
source_base_dir=../data/workbench/Extracted
target_base_dir=../output/SubtreeElement
pretrained_model=microsoft/codebert-base

python3 ../main.py \
    --lang $lang \
    --target_subtree_root if_statement elif_clause else_clause for_statement while_statement expression_statement return_statement break_statement with_statement \
    --source_base_dir $source_base_dir \
    --target_base_dir $target_base_dir \
    --upper_data_size 500 

