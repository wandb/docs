#!/bin/usr
subdir="ko"

# Clean build directory
yarn clear

# Build the docs and store them in the correct build subdir
yarn docusaurus build --out-dir build/$subdir --config docusaurus.config.js

echo "Formatting $language docs for GAE..."

cp -R ./build/$subdir/img ./build/
cp ./build/$subdir/index.html ./build/
