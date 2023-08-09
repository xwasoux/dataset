dir_name=projectName

mkdir $dir_name
dir_path=$(pwd)

cd $dir_name
mkdir data data/workbench output sh

touch preprocess.py main.py

cd sh/
touch preprocess.sh main.sh

cd $dir_path