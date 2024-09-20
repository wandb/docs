rm -rf ../build
git clone https://github.com/wandb/docodile
cd docodile
git checkout japanese_docs
bash ./scripts/build-prod-docs.sh
mv build ../build-ja
git checkout korean_docs
bash ./scripts/build-prod-docs.sh
mv build ../build-ko
git checkout main
yarn build:prod
mv build ../build-en
cd ..
rm -rf docodile
mv build-ja/ja build-en/ja
mv build-ko/ko build-en/ko
mv build-en ../build