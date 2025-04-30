# Run this from root by calling sh ./scripts/production.sh

# Build the English docs from this branch

# Update Hugo if necessary
hugo mod get -u
# Clear out previous build artifacts
rm -rf public

# Detect whether we are in Cloudflare
if [ -n "$CF_PAGES" ] && [ -n "$CF_PAGES_URL" ]; then
    echo "Building in Cloudflare with these variables:"
    echo "  CF_PAGES: $CF_PAGES"
    echo "  CF_PAGES_URL: $CF_PAGES_URL"
    echo "  CF_PAGES_COMMIT_SHA: $CF_PAGES_COMMIT_SHA"
    echo "  CF_PAGES_BRANCH: $CF_PAGES_BRANCH"
    if [ "$CF_PAGES_BRANCH" = "main" ]; then
        echo "Building the main site"
        hugo
    else
        echo "Building a PR preview"
        hugo -b $CF_PAGES_URL
    fi
else
    echo "Building locally"
    hugo
fi

# Move English sitemap into root
rm public/sitemap.xml
mv public/en/sitemap.xml public/

# Clear Hugo translated content, which we don't yet use
rm -rf public/ja
rm -rf public/ko

# Use a secondary git clone to build JA and KO 
rm -rf scripts/docs
git clone --depth 1 git@github.com:wandb/docs.git scripts/docs
cd scripts/docs
git remote set-branches origin '*'
git fetch -v --depth=1

# The JA and KO builds use special scripts for the baseURL mod
# JA (requires node 18.0.0)
git checkout japanese_docs
npm -g install yarn
yarn install
sh scripts/build-prod-docs.sh
mv build/ja ../../public/ja
git stash
# KO
git checkout korean_docs
npm -g install yarn
yarn install
sh scripts/build-prod-docs.sh
mv build/ko ../../public/ko
git stash
# Cleanup
cd ../..
rm -rf scripts/docs
