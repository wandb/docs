#!/bin/bash

# JIRA API Configuration
JIRA_HOST="https://wandb.atlassian.net"
JIRA_USERNAME="noah.luna@wandb.com"
# Source .zshenv to get JIRA_API_TOKEN if not already set
if [ -z "$JIRA_API_TOKEN" ]; then
    if [ -f "$HOME/.zshenv" ]; then
        source "$HOME/.zshenv"
    fi
fi

# Check if JIRA_API_TOKEN is set
if [ -z "$JIRA_API_TOKEN" ]; then
    echo "Error: JIRA_API_TOKEN not found in environment"
    echo "Please export JIRA_API_TOKEN in your .zshenv file"
    exit 1
fi

# Base64 encode credentials for Basic Auth
AUTH_HEADER=$(echo -n "$JIRA_USERNAME:$JIRA_API_TOKEN" | base64)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to make authenticated JIRA API requests
jira_request() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    if [ -z "$data" ]; then
        curl -s -X "$method" \
            -H "Authorization: Basic $AUTH_HEADER" \
            -H "Accept: application/json" \
            -H "Content-Type: application/json" \
            "$JIRA_HOST/rest/api/3/$endpoint"
    else
        curl -s -X "$method" \
            -H "Authorization: Basic $AUTH_HEADER" \
            -H "Accept: application/json" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$JIRA_HOST/rest/api/3/$endpoint"
    fi
}

# List all projects
list_projects() {
    echo -e "${BLUE}Fetching all projects...${NC}"
    jira_request GET "project" | jq -r '.[] | "\(.key): \(.name)"'
}

# Get issue details
get_issue() {
    local issue_key=$1
    if [ -z "$issue_key" ]; then
        echo -e "${RED}Error: Issue key required${NC}"
        echo "Usage: $0 get-issue <ISSUE-KEY>"
        return 1
    fi
    
    echo -e "${BLUE}Fetching issue $issue_key...${NC}"
    jira_request GET "issue/$issue_key" | jq '.'
}

# Create new issue
create_issue() {
    local project_key=$1
    local summary=$2
    local description=$3
    local issue_type=${4:-"Task"}
    
    if [ -z "$project_key" ] || [ -z "$summary" ]; then
        echo -e "${RED}Error: Project key and summary required${NC}"
        echo "Usage: $0 create-issue <PROJECT-KEY> <SUMMARY> [DESCRIPTION] [ISSUE-TYPE]"
        return 1
    fi
    
    local data=$(cat <<EOF
{
    "fields": {
        "project": {
            "key": "$project_key"
        },
        "summary": "$summary",
        "description": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": "$description"
                        }
                    ]
                }
            ]
        },
        "issuetype": {
            "name": "$issue_type"
        }
    }
}
EOF
)
    
    echo -e "${BLUE}Creating new issue...${NC}"
    jira_request POST "issue" "$data" | jq -r '.key'
}

# Update issue
update_issue() {
    local issue_key=$1
    local field=$2
    local value=$3
    
    if [ -z "$issue_key" ] || [ -z "$field" ] || [ -z "$value" ]; then
        echo -e "${RED}Error: Issue key, field, and value required${NC}"
        echo "Usage: $0 update-issue <ISSUE-KEY> <FIELD> <VALUE>"
        return 1
    fi
    
    local data=""
    case $field in
        "summary")
            data='{"fields": {"summary": "'"$value"'"}}'
            ;;
        "description")
            data='{"fields": {"description": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": "'"$value"'"}]}]}}}'
            ;;
        "status")
            # For status updates, we need to use transitions
            echo -e "${BLUE}Getting available transitions...${NC}"
            local transitions=$(jira_request GET "issue/$issue_key/transitions")
            echo "$transitions" | jq -r '.transitions[] | "\(.id): \(.name)"'
            echo -e "${BLUE}Use: $0 transition-issue <ISSUE-KEY> <TRANSITION-ID>${NC}"
            return 0
            ;;
        *)
            data='{"fields": {"'"$field"'": "'"$value"'"}}'
            ;;
    esac
    
    echo -e "${BLUE}Updating issue $issue_key...${NC}"
    jira_request PUT "issue/$issue_key" "$data"
    echo -e "${GREEN}Issue updated successfully${NC}"
}

# Transition issue (change status)
transition_issue() {
    local issue_key=$1
    local transition_id=$2
    
    if [ -z "$issue_key" ] || [ -z "$transition_id" ]; then
        echo -e "${RED}Error: Issue key and transition ID required${NC}"
        echo "Usage: $0 transition-issue <ISSUE-KEY> <TRANSITION-ID>"
        return 1
    fi
    
    local data='{"transition": {"id": "'"$transition_id"'"}}'
    
    echo -e "${BLUE}Transitioning issue $issue_key...${NC}"
    jira_request POST "issue/$issue_key/transitions" "$data"
    echo -e "${GREEN}Issue transitioned successfully${NC}"
}

# Search issues with JQL
search_issues() {
    local jql=$1
    local max_results=${2:-50}
    
    if [ -z "$jql" ]; then
        echo -e "${RED}Error: JQL query required${NC}"
        echo "Usage: $0 search <JQL-QUERY> [MAX-RESULTS]"
        echo "Example: $0 search 'project = PROJ AND status = Open' 10"
        return 1
    fi
    
    # URL encode the JQL
    local encoded_jql=$(echo -n "$jql" | jq -sRr @uri)
    
    echo -e "${BLUE}Searching with JQL: $jql${NC}"
    jira_request GET "search?jql=$encoded_jql&maxResults=$max_results" | \
        jq -r '.issues[] | "\(.key): \(.fields.summary) [\(.fields.status.name)]"'
}

# Add comment to issue
add_comment() {
    local issue_key=$1
    local comment=$2
    
    if [ -z "$issue_key" ] || [ -z "$comment" ]; then
        echo -e "${RED}Error: Issue key and comment required${NC}"
        echo "Usage: $0 add-comment <ISSUE-KEY> <COMMENT>"
        return 1
    fi
    
    local data=$(cat <<EOF
{
    "body": {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": "$comment"
                    }
                ]
            }
        ]
    }
}
EOF
)
    
    echo -e "${BLUE}Adding comment to $issue_key...${NC}"
    jira_request POST "issue/$issue_key/comment" "$data" | jq -r '.id'
    echo -e "${GREEN}Comment added successfully${NC}"
}

# Get my open issues
my_issues() {
    echo -e "${BLUE}Fetching your open issues...${NC}"
    search_issues "assignee = currentUser() AND resolution = Unresolved ORDER BY priority DESC, updated DESC"
}

# Function to create JIRA issue and trigger GitHub PR
create_issue_with_pr() {
    local project_key=$1
    local summary=$2
    local description=$3
    local github_repo=$4
    local branch_name=$5
    local pr_base=${6:-"main"}
    
    if [ -z "$project_key" ] || [ -z "$summary" ] || [ -z "$github_repo" ] || [ -z "$branch_name" ]; then
        echo -e "${RED}Error: Missing required parameters${NC}"
        echo "Usage: $0 create-issue-pr <PROJECT> <SUMMARY> <DESC> <REPO> <BRANCH> [BASE]"
        return 1
    fi
    
    # Create JIRA issue first
    echo -e "${BLUE}Creating JIRA issue...${NC}"
    local jira_key=$(create_issue "$project_key" "$summary" "$description" "Task")
    
    if [ $? -eq 0 ] && [ -n "$jira_key" ]; then
        echo -e "${GREEN}Created JIRA issue: $jira_key${NC}"
        
        # Call GitHub API script to create PR
        echo -e "${BLUE}Creating GitHub PR...${NC}"
        local pr_title="[$jira_key] $summary"
        local pr_body="## JIRA Issue
[View in JIRA]($JIRA_HOST/browse/$jira_key)

## Description
$description

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [x] Documentation update
- [ ] Other (please specify)"
        
        # Execute github-api.sh script
        "$(dirname "$0")/github-api.sh" create-pr "$github_repo" "$pr_title" "$branch_name" "$pr_base" "$pr_body"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully created JIRA issue and GitHub PR!${NC}"
            # Add comment to JIRA with PR link (you'd need to capture PR URL)
            add_comment "$jira_key" "GitHub PR created for this issue"
        fi
    else
        echo -e "${RED}Failed to create JIRA issue${NC}"
        return 1
    fi
}

# Main command handler
case "$1" in
    "list-projects")
        list_projects
        ;;
    "get-issue")
        get_issue "$2"
        ;;
    "create-issue")
        create_issue "$2" "$3" "$4" "$5"
        ;;
    "update-issue")
        update_issue "$2" "$3" "$4"
        ;;
    "transition-issue")
        transition_issue "$2" "$3"
        ;;
    "search")
        search_issues "$2" "$3"
        ;;
    "add-comment")
        add_comment "$2" "$3"
        ;;
    "my-issues")
        my_issues
        ;;
    "create-issue-pr")
        create_issue_with_pr "$2" "$3" "$4" "$5" "$6" "$7"
        ;;
    *)
        echo "JIRA API Command Line Tool"
        echo ""
        echo "Usage: $0 <command> [arguments]"
        echo ""
        echo "Commands:"
        echo "  list-projects                    - List all projects"
        echo "  get-issue <KEY>                  - Get issue details"
        echo "  create-issue <PROJ> <SUMMARY> [DESC] [TYPE] - Create new issue"
        echo "  update-issue <KEY> <FIELD> <VALUE> - Update issue field"
        echo "  transition-issue <KEY> <ID>      - Change issue status"
        echo "  search <JQL> [MAX]               - Search with JQL query"
        echo "  add-comment <KEY> <COMMENT>      - Add comment to issue"
        echo "  my-issues                        - List your open issues"
        echo "  create-issue-pr <PROJ> <SUMMARY> <DESC> <REPO> <BRANCH> [BASE] - Create JIRA issue and GitHub PR"
        echo ""
        echo "Examples:"
        echo "  $0 list-projects"
        echo "  $0 get-issue PROJ-123"
        echo "  $0 create-issue PROJ \"Fix login bug\" \"Users cannot login\" \"Bug\""
        echo "  $0 search 'project = PROJ AND status = \"In Progress\"'"
        echo "  $0 add-comment PROJ-123 \"Working on this now\""
        echo "  $0 create-issue-pr PROJ \"Update API docs\" \"Add examples\" owner/repo feature-branch"
        ;;
esac