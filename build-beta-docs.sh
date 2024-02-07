#!/bin/usr
subdir="ko"

# Build the docs and store them in the correct build subdir
docusaurus build --out-dir build/$subdir --config docusaurus-beta.config.js

echo "Formatting $language docs for GAE..."
cd ./build/$subdir
cp -R ./img ../
mv index.html ../