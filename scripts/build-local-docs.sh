#!/bin/usr
subdir="ja"

# Build the docs and store them in the correct build subdir
yarn docusaurus build --out-dir build/$subdir

echo "Formatting $language docs for GAE..."

cp -R ./build/$subdir/img ./build/
cp ./build/$subdir/index.html ./build/
