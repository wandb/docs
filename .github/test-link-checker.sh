#!/bin/bash
set -e

echo "=== Link Checker Local Test ==="
echo "This script simulates the link checker workflow locally"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Create test content with broken links
echo -e "${YELLOW}Step 1: Creating test content...${NC}"
mkdir -p content/test-broken-links/{en,ja}/ref/{js,python}

# Create test files with various broken links
cat > content/test-broken-links/broken-links.md <<'EOF'
# Test Broken Links

Here are some test links:

1. HTTP that should be HTTPS: http://example.com
2. Missing trailing slash: https://docs.wandb.ai/guides
3. With trailing slash that might redirect: https://example.com/path/
4. A working link: https://google.com
5. An actually broken link: https://example.com/this-page-does-not-exist-404
6. Link with www: http://www.example.com
7. GitHub link: http://github.com/wandb/docs

EOF

# Create a file in generated reference docs (should be skipped)
cat > content/test-broken-links/en/ref/python/test-generated.md <<'EOF'
# Generated Python Reference

This file should be skipped:
- http://example.com
- http://github.com/test

EOF

# Create a file in translated content (should NOT be skipped)
cat > content/test-broken-links/ja/translated.md <<'EOF'
# Japanese Content

This should be processed:
- http://example.com
- https://docs.wandb.ai/guides/

EOF

# Step 2: Run lychee to check links
echo -e "${YELLOW}Step 2: Running lychee to check links...${NC}"
mkdir -p lychee

# Create a simulated lychee output
cat > lychee/out.json <<'EOF'
{
  "fail_map": {
    "http://example.com": [
      {"input": "content/test-broken-links/broken-links.md", "line": 5},
      {"input": "content/test-broken-links/en/ref/python/test-generated.md", "line": 4},
      {"input": "content/test-broken-links/ja/translated.md", "line": 4}
    ],
    "https://docs.wandb.ai/guides": [
      {"input": "content/test-broken-links/broken-links.md", "line": 6},
      {"input": "content/test-broken-links/ja/translated.md", "line": 5}
    ],
    "https://example.com/path/": [
      {"input": "content/test-broken-links/broken-links.md", "line": 7}
    ],
    "https://example.com/this-page-does-not-exist-404": [
      {"input": "content/test-broken-links/broken-links.md", "line": 9}
    ],
    "http://www.example.com": [
      {"input": "content/test-broken-links/broken-links.md", "line": 10}
    ],
    "http://github.com/wandb/docs": [
      {"input": "content/test-broken-links/broken-links.md", "line": 11}
    ],
    "http://github.com/test": [
      {"input": "content/test-broken-links/en/ref/python/test-generated.md", "line": 5}
    ]
  }
}
EOF

# Step 3: Run the fix script
echo -e "${YELLOW}Step 3: Running fix_broken_links.mjs...${NC}"
node .github/scripts/fix_broken_links.mjs

# Step 4: Show the results
echo ""
echo -e "${GREEN}Step 4: Results${NC}"
echo "================================"

if [ -f .github/lychee-report.md ]; then
    echo -e "${YELLOW}Generated report:${NC}"
    cat .github/lychee-report.md
    echo ""
fi

# Show what files were modified
echo -e "${YELLOW}Modified files:${NC}"
git diff --name-only content/test-broken-links/ 2>/dev/null || echo "No git repository, showing file contents instead:"

# Show the actual changes
echo ""
echo -e "${YELLOW}Changes made:${NC}"
for file in content/test-broken-links/broken-links.md content/test-broken-links/ja/translated.md; do
    if [ -f "$file" ]; then
        echo "--- $file ---"
        cat "$file" | grep -E "https?://" || true
        echo ""
    fi
done

# Check if generated docs were skipped
echo -e "${YELLOW}Checking generated docs (should be unchanged):${NC}"
if grep -q "http://example.com" content/test-broken-links/en/ref/python/test-generated.md; then
    echo -e "${GREEN}✓ Generated Python reference docs were correctly skipped${NC}"
else
    echo -e "${RED}✗ Generated Python reference docs were incorrectly modified${NC}"
fi

# Cleanup
echo ""
echo -e "${YELLOW}Step 5: Cleanup${NC}"
echo "To clean up test files, run: rm -rf content/test-broken-links lychee/out.json .github/lychee-report.md"
echo ""
echo -e "${GREEN}Test complete!${NC}"