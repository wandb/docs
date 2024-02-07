#!/bin/usr
language="Korean"

yarn docusaurus build --out-dir build/ko

echo "Formatting $language docs for GAE..."
cd ./build/ko
cp -R ./img ../
mv index.html ../

echo "Done!" 