# Run this from root by calling sh ./scripts/production.sh

# Build the English docs from this branch

# Read Hugo version from wrangler.toml
CLOUDFLARE_HUGO_VERSION="$(grep 'HUGO_VERSION' wrangler.toml | awk -F '\"' {'print $2'})"

# Update Hugo if necessary
hugo mod get -u
# Clear out previous build artifacts
rm -rf public

# Detect whether we are in Cloudflare
if [ -n "$CF_PAGES" ] && [ -n "$CF_PAGES_URL" ]; then
    echo "Building in Cloudflare with these variables:"
    echo "  HUGO_VERSION from wrangler.toml: $CLOUDFLARE_HUGO_VERSION"
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
    echo "Building locally with Hugo version $(hugo version)"
    hugo
fi

# Move English sitemap into root
rm public/sitemap.xml
mv public/en/sitemap.xml public/