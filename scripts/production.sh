# Run this from root by calling sh ./scripts/production.sh

# Build the English docs from this branch
hugo mod get -u
if [ -n "$CF_PAGES_URL" ]; then
    echo "Building in Cloudflare"
    hugo -b $CF_PAGES_URL
else
    echo "Building locally"
    hugo
fi