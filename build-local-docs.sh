#!/bin/usr
subdir="ko"

# Build the docs and store them in the correct build subdir
yarn docusaurus build --out-dir build/$subdir

echo "Formatting $language docs for GAE..."
cd ./build/$subdir
cp -R ./img ../
mv index.html ../