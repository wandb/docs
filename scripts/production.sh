#!/bin/bash
if [ "$CF_PAGES_BRANCH" == "staging" ]; then
    yarn build:prod
else
    rm -rf ../build
    git config --global http.version HTTP/1.1
    git clone --depth 1 https://github.com/wandb/docodile ./docodile
    cd docodile
    git remote set-branches origin '*'
    git fetch -v --depth=1
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
fi