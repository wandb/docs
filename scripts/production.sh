#!/bin/bash
git fetch
rm -rf build
yarn build:prod
mv build ..
git checkout japanese_docs
bash ./scripts/build-prod-docs.sh
mv build ../build/ja
git checkout korean_docs
bash ./scripts/build-prod-docs.sh
mv build/ko ../build
rm -rf build
mv ../build .