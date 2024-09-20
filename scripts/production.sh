#!/bin/bash
if [ "$CF_PAGES_BRANCH" == "staging" ]; then
    yarn build:prod
else
    cd ../
    rm -rf build
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
    mv build-ja/ja build-en/ja
    mv build-ko/ko build-en/ko
    mv build-en ../build
fi