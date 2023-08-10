CURRENT=$(pwd)
dataset_name=SubtreeEmelent

mkdir ../data ../data/workbench ../data/workbench/Extracted
mkdir ../output/$dataset_name
cd ../data/workbench
gdown 1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip

cd $CURRENT

lang=python
source_base_dir=../data/workbench/CodeSearchNet
target_base_dir=../data/workbench/Extracted
pretrained_model=microsoft/codebert-base

python3 ../preprocess.py \
    --lang $lang \
    --source_base_dir $source_base_dir \
    --target_base_dir $target_base_dir \
    --pretrained_model $pretrained_model


