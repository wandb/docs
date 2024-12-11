# Run this from root by calling sh ./scripts/production.sh

# Build the English docs from this branch
go mod init docsy
hugo mod get -u
hugo

# Use a secondary git clone to build JA and KO 
rm -rf scripts/docodile
git clone --depth 1 git@github.com:wandb/docodile.git scripts/docodile
cd scripts/docodile
git remote set-branches origin '*'
git fetch -v --depth=1

# The JA and KO builds use special scripts for the baseURL mod
# JA (requires node 18.0.0)
git checkout japanese_docs
yarn install
sh scripts/build-prod-docs.sh
mv build/ja ../../public
git stash
# KO
git checkout korean_docs
yarn install
sh scripts/build-prod-docs.sh
mv build/ko ../../public
git stash
# Cleanup
cd ../..
rm -rf scripts/docodile