cd ..
mkdir zip

target_path=diff-commit/wakame-tech/12_step_emb_os
zip_name=wakame-tech/12_step_emb_os.zip

zip -r zip/zipjsonl $target_path

cd sh/