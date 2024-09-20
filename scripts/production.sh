#!/bin/bash
if [ "$CF_PAGES_BRANCH" == "staging" ]; then
    yarn build:prod
else
    cd ../
    rm -rf build
    yarn build:prod
    mv build ..
    git checkout japanese_docs
    bash ./scripts/build-prod-docs.sh
    mv build ../build/ja
    git checkout korean_docs
    bash ./scripts/build-prod-docs.sh
    mv build/ko ../build
fi