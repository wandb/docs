#!/bin/bash
set -e

LOGFILE="/tmp/mint-dev-$$.log"
PARSE_TIME=45  # Give Mintlify time to log parsing errors
PID=""

# Cleanup function
cleanup() {
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    sleep 0.5
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f "$LOGFILE"
}

# Trap to ensure cleanup
trap cleanup EXIT INT TERM

# Check if there are any MDX files in the changeset
if [ -n "$GITHUB_BASE_REF" ]; then
  # In a PR context, check changed files
  echo "Checking for changed MDX files in PR..."
  
  # Get the list of changed files
  CHANGED_MDX=$(git diff --name-only "origin/$GITHUB_BASE_REF"...HEAD -- '*.mdx' 2>/dev/null | grep -E '\.mdx$' || true)
  
  if [ -z "$CHANGED_MDX" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "NO MDX FILES CHANGED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "No MDX files found in this PR. Skipping validation."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
  fi
  
  echo "Found changed MDX files:"
  echo "$CHANGED_MDX" | sed 's/^/  /'
  echo ""
fi

echo "Starting Mintlify validation..."
echo ""
echo "Running: mint dev --no-open (will run for ${PARSE_TIME}s to parse all files)"
echo ""

# Run mint dev with tee to force output writing, timeout after PARSE_TIME seconds
# Use timeout if available (Linux), otherwise use gtimeout (macOS with coreutils), or perl as fallback
if command -v timeout > /dev/null 2>&1; then
  timeout --preserve-status ${PARSE_TIME}s mint dev --no-open 2>&1 | tee "$LOGFILE" > /dev/null || true
elif command -v gtimeout > /dev/null 2>&1; then
  gtimeout --preserve-status ${PARSE_TIME}s mint dev --no-open 2>&1 | tee "$LOGFILE" > /dev/null || true
else
  # Fallback: run mint dev in background and kill after PARSE_TIME
  mint dev --no-open 2>&1 | tee "$LOGFILE" > /dev/null &
  PID=$!
  sleep ${PARSE_TIME}
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
fi

echo ""
echo "✓ Mintlify finished parsing"
echo ""

# Check for parsing errors
if grep -q "parsing error" "$LOGFILE"; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "❌ MINTLIFY PARSING ERRORS DETECTED"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Parsing errors found:"
  echo ""
  grep "parsing error" "$LOGFILE" | sed 's/^/  /'
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "💡 These are Mintlify parsing errors. Please fix them or"
  echo "   file an issue if you believe they are incorrect:"
  echo "   https://github.com/wandb/docs/issues/new"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ MINTLIFY PARSING VALIDATION PASSED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "No parsing errors detected by Mintlify"
echo ""

# Run broken links check
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CHECKING FOR BROKEN LINKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: mint broken-links"
echo ""

# Run mint broken-links - it will exit with non-zero if broken links are found
mint broken-links

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL VALIDATION CHECKS PASSED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "- No parsing errors"
echo "- No broken links"
exit 0
