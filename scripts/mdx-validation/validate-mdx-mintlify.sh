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

echo "Starting Mintlify validation..."
echo ""
echo "Running: mint dev --no-open (will run for ${PARSE_TIME}s to parse all files)"
echo ""

# Run mint dev with tee to force output writing, timeout after PARSE_TIME seconds
timeout --preserve-status ${PARSE_TIME}s mint dev --no-open 2>&1 | tee "$LOGFILE" > /dev/null || true

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
else
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✅ MINTLIFY VALIDATION PASSED"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "No parsing errors detected by Mintlify"
  exit 0
fi

