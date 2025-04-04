# Run this from root by calling sh ./scripts/production.sh

# Build the English docs from this branch
hugo mod get -u
rm -rf public
hugo
rm public/sitemap.xml
mv public/en/sitemap.xml public/sitemap.xml