CURRENT=$(cd $(dirname $0);pwd)

mkdir ../data ../data/deploy ../data/astcutting
cd ../data/deploy
# gdown 1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
# unzip Cleaned_CodeSearchNet.zip
# rm Cleaned_CodeSearchNet.zip

cd $CURRENT

lang=python
basedir=../data/deploy/CodeSearchNet

python3 ../src/astcutting.py \
    --language $lang \
    --basedir $basedir
