# Run this from root by calling sh ./scripts/production.sh

# Build the English docs from this branch
yarn build:prod

# Use a secondary git clone to build JA and KO 
rm -rf scripts/docodile
git clone --depth 1 git@github.com:wandb/docodile.git scripts/docodile
cd scripts/docodile
git remote set-branches origin '*'
git fetch -v --depth=1

# The JA and KO builds use special scripts for the baseURL mod
# JA
git checkout japanese_docs
NODE_VERSION="18.0.0"
yarn install
node --version
sh scripts/build-prod-docs.sh
mv build/ja ../../build
# KO
git checkout korean_docs
NODE_VERSION="18.0.0"
yarn install
node --version
sh scripts/build-prod-docs.sh
mv build/ko ../../build

# Cleanup
cd ../..
rm -rf scripts/docodile