CURRENT=$(pwd)

mkdir ../data ../data/workbench ../data/astcutting
cd ../data/workbench
gdown 1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip

cd $CURRENT

lang=python
source_base_dir=../data/workbench/CodeSearchNet
target_base_dir=../data/astcutting

python3 ../src/astcutting.py \
    --lang $lang \
    --source_base_dir $source_base_dir \
    --target_base_dir $target_base_dir
