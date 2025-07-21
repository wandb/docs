PROMPT_FILE="$1"
JIRA_TICKET_NUMBER="$2"

# --- Input validation ---
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <prompt_file> <jira_ticket_number>"
  echo "Example: $0 prompt.txt 'Fix the bug in the login flow'"
  exit 1
fi

bash ./scripts/helper_scripts/jira-api.sh get-issue $JIRA_TICKET_NUMBER |  sed  '1d' | sed -r "s/\x1B\[[0-9;]*[mK]//g" > jira_ticket_info.json

echo -e "$(cat "$PROMPT_FILE")\n\n$(cat jira_ticket_info.json)" | claude