#!/bin/bash
rm -rf scripts/docodile
git clone https://github.com/wandb/docodile.git scripts/docodile
cd scripts/docodile
git fetch
yarn build:prod
mv build ../..
git checkout japanese_docs
bash ./scripts/build-prod-docs.sh
mv build/ja ../../build
git checkout korean_docs
bash ./scripts/build-prod-docs.sh
mv build/ko ../../build