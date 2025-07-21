#!/bin/bash

# GitHub API Configuration
GITHUB_API_URL="https://api.github.com"
# Set your GitHub personal access token here or export as environment variable
GITHUB_TOKEN="INSERT_YOUR_GITHUB_TOKEN"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if token is set
check_token() {
    if [ -z "$GITHUB_TOKEN" ]; then
        echo -e "${RED}Error: GitHub token not set${NC}"
        echo "Please set GITHUB_TOKEN environment variable or edit this script"
        echo "Get a token from: https://github.com/settings/tokens"
        exit 1
    fi
}

# Function to make authenticated GitHub API requests
github_request() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    if [ -z "$data" ]; then
        curl -s -X "$method" \
            -H "Authorization: Bearer $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "$GITHUB_API_URL/$endpoint"
    else
        curl -s -X "$method" \
            -H "Authorization: Bearer $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$GITHUB_API_URL/$endpoint"
    fi
}

# Get authenticated user info
whoami() {
    echo -e "${BLUE}Getting authenticated user info...${NC}"
    github_request GET "user" | jq -r '. | "Username: \(.login)\nName: \(.name)\nEmail: \(.email)\nPublic Repos: \(.public_repos)\nFollowers: \(.followers)"'
}

# List user repositories
list_repos() {
    local user=${1:-}
    local visibility=${2:-"all"}  # all, public, private
    local per_page=${3:-30}
    
    if [ -z "$user" ]; then
        echo -e "${BLUE}Fetching your repositories...${NC}"
        github_request GET "user/repos?visibility=$visibility&per_page=$per_page&sort=updated" | \
            jq -r '.[] | "\(.full_name) [\(.private | if . then "private" else "public" end)] - \(.description // "No description")"'
    else
        echo -e "${BLUE}Fetching repositories for $user...${NC}"
        github_request GET "users/$user/repos?type=all&per_page=$per_page&sort=updated" | \
            jq -r '.[] | "\(.full_name) - \(.description // "No description")"'
    fi
}

# Get repository details
get_repo() {
    local repo=$1
    
    if [ -z "$repo" ]; then
        echo -e "${RED}Error: Repository required${NC}"
        echo "Usage: $0 get-repo <OWNER/REPO>"
        return 1
    fi
    
    echo -e "${BLUE}Fetching repository $repo...${NC}"
    github_request GET "repos/$repo" | jq '.'
}

# Create new repository
create_repo() {
    local name=$1
    local description=${2:-""}
    local private=${3:-false}
    local auto_init=${4:-true}
    
    if [ -z "$name" ]; then
        echo -e "${RED}Error: Repository name required${NC}"
        echo "Usage: $0 create-repo <NAME> [DESCRIPTION] [PRIVATE:true/false] [AUTO_INIT:true/false]"
        return 1
    fi
    
    local data=$(cat <<EOF
{
    "name": "$name",
    "description": "$description",
    "private": $private,
    "auto_init": $auto_init
}
EOF
)
    
    echo -e "${BLUE}Creating repository $name...${NC}"
    github_request POST "user/repos" "$data" | jq -r '. | "Created: \(.full_name)\nURL: \(.html_url)\nClone: \(.clone_url)"'
}

# List issues
list_issues() {
    local repo=$1
    local state=${2:-"open"}  # open, closed, all
    local assignee=${3:-""}
    
    if [ -z "$repo" ]; then
        echo -e "${RED}Error: Repository required${NC}"
        echo "Usage: $0 list-issues <OWNER/REPO> [STATE] [ASSIGNEE]"
        return 1
    fi
    
    local query="state=$state"
    if [ -n "$assignee" ]; then
        query="$query&assignee=$assignee"
    fi
    
    echo -e "${BLUE}Fetching issues for $repo...${NC}"
    github_request GET "repos/$repo/issues?$query" | \
        jq -r '.[] | "#\(.number): \(.title) [\(.state)] by @\(.user.login)"'
}

# Create new issue
create_issue() {
    local repo=$1
    local title=$2
    local body=${3:-""}
    local labels=${4:-""}
    
    if [ -z "$repo" ] || [ -z "$title" ]; then
        echo -e "${RED}Error: Repository and title required${NC}"
        echo "Usage: $0 create-issue <OWNER/REPO> <TITLE> [BODY] [LABELS]"
        return 1
    fi
    
    local labels_json="[]"
    if [ -n "$labels" ]; then
        labels_json=$(echo "$labels" | tr ',' '\n' | jq -R . | jq -s .)
    fi
    
    local data=$(cat <<EOF
{
    "title": "$title",
    "body": "$body",
    "labels": $labels_json
}
EOF
)
    
    echo -e "${BLUE}Creating issue in $repo...${NC}"
    github_request POST "repos/$repo/issues" "$data" | \
        jq -r '. | "Created issue #\(.number): \(.title)\nURL: \(.html_url)"'
}

# Update issue
update_issue() {
    local repo=$1
    local number=$2
    local field=$3
    local value=$4
    
    if [ -z "$repo" ] || [ -z "$number" ] || [ -z "$field" ] || [ -z "$value" ]; then
        echo -e "${RED}Error: Repository, issue number, field, and value required${NC}"
        echo "Usage: $0 update-issue <OWNER/REPO> <NUMBER> <FIELD> <VALUE>"
        echo "Fields: title, body, state, labels"
        return 1
    fi
    
    local data=""
    case $field in
        "title"|"body"|"state")
            data='{"'$field'": "'"$value"'"}'
            ;;
        "labels")
            local labels_json=$(echo "$value" | tr ',' '\n' | jq -R . | jq -s .)
            data='{"labels": '$labels_json'}'
            ;;
        *)
            echo -e "${RED}Unknown field: $field${NC}"
            return 1
            ;;
    esac
    
    echo -e "${BLUE}Updating issue #$number in $repo...${NC}"
    github_request PATCH "repos/$repo/issues/$number" "$data"
    echo -e "${GREEN}Issue updated successfully${NC}"
}

# List pull requests
list_prs() {
    local repo=$1
    local state=${2:-"open"}  # open, closed, all
    
    if [ -z "$repo" ]; then
        echo -e "${RED}Error: Repository required${NC}"
        echo "Usage: $0 list-prs <OWNER/REPO> [STATE]"
        return 1
    fi
    
    echo -e "${BLUE}Fetching pull requests for $repo...${NC}"
    github_request GET "repos/$repo/pulls?state=$state" | \
        jq -r '.[] | "#\(.number): \(.title) [\(.state)] by @\(.user.login) - \(.head.ref) → \(.base.ref)"'
}

# Create pull request
create_pr() {
    local repo=$1
    local title=$2
    local head=$3
    local base=${4:-"main"}
    local body=${5:-""}
    
    if [ -z "$repo" ] || [ -z "$title" ] || [ -z "$head" ]; then
        echo -e "${RED}Error: Repository, title, and head branch required${NC}"
        echo "Usage: $0 create-pr <OWNER/REPO> <TITLE> <HEAD> [BASE] [BODY]"
        return 1
    fi
    
    local data=$(cat <<EOF
{
    "title": "$title",
    "head": "$head",
    "base": "$base",
    "body": "$body"
}
EOF
)
    
    echo -e "${BLUE}Creating pull request in $repo...${NC}"
    github_request POST "repos/$repo/pulls" "$data" | \
        jq -r '. | "Created PR #\(.number): \(.title)\nURL: \(.html_url)"'
}

# Search repositories
search_repos() {
    local query=$1
    local sort=${2:-"stars"}  # stars, forks, updated
    local per_page=${3:-10}
    
    if [ -z "$query" ]; then
        echo -e "${RED}Error: Search query required${NC}"
        echo "Usage: $0 search-repos <QUERY> [SORT] [LIMIT]"
        return 1
    fi
    
    local encoded_query=$(echo -n "$query" | jq -sRr @uri)
    
    echo -e "${BLUE}Searching repositories: $query${NC}"
    github_request GET "search/repositories?q=$encoded_query&sort=$sort&per_page=$per_page" | \
        jq -r '.items[] | "\(.full_name) ⭐ \(.stargazers_count) - \(.description // "No description")"'
}

# Search issues/PRs
search_issues() {
    local query=$1
    local sort=${2:-"created"}  # created, updated, comments
    local per_page=${3:-10}
    
    if [ -z "$query" ]; then
        echo -e "${RED}Error: Search query required${NC}"
        echo "Usage: $0 search-issues <QUERY> [SORT] [LIMIT]"
        echo "Example: $0 search-issues 'repo:owner/name is:issue is:open'"
        return 1
    fi
    
    local encoded_query=$(echo -n "$query" | jq -sRr @uri)
    
    echo -e "${BLUE}Searching issues: $query${NC}"
    github_request GET "search/issues?q=$encoded_query&sort=$sort&per_page=$per_page" | \
        jq -r '.items[] | "\(.repository_url | split("/") | .[-2]+"/"+.[-1]) #\(.number): \(.title) [\(.state)]"'
}

# Get notifications
get_notifications() {
    local all=${1:-false}
    
    echo -e "${BLUE}Fetching notifications...${NC}"
    github_request GET "notifications?all=$all" | \
        jq -r '.[] | "\(.repository.full_name): \(.subject.title) [\(.subject.type)] - \(.reason)"'
}

# Star a repository
star_repo() {
    local repo=$1
    
    if [ -z "$repo" ]; then
        echo -e "${RED}Error: Repository required${NC}"
        echo "Usage: $0 star <OWNER/REPO>"
        return 1
    fi
    
    echo -e "${BLUE}Starring $repo...${NC}"
    github_request PUT "user/starred/$repo" ""
    echo -e "${GREEN}Repository starred${NC}"
}

# Create gist
create_gist() {
    local description=$1
    local filename=$2
    local content=$3
    local public=${4:-false}
    
    if [ -z "$description" ] || [ -z "$filename" ]; then
        echo -e "${RED}Error: Description and filename required${NC}"
        echo "Usage: $0 create-gist <DESCRIPTION> <FILENAME> [CONTENT|-] [PUBLIC:true/false]"
        echo "Use - to read content from stdin"
        return 1
    fi
    
    if [ "$content" = "-" ] || [ -z "$content" ]; then
        content=$(cat)
    fi
    
    local data=$(cat <<EOF
{
    "description": "$description",
    "public": $public,
    "files": {
        "$filename": {
            "content": $(echo "$content" | jq -Rs .)
        }
    }
}
EOF
)
    
    echo -e "${BLUE}Creating gist...${NC}"
    github_request POST "gists" "$data" | \
        jq -r '. | "Created gist: \(.description)\nURL: \(.html_url)\nRaw: \(.files | to_entries[0].value.raw_url)"'
}

# Main command handler
check_token

case "$1" in
    "whoami")
        whoami
        ;;
    "list-repos")
        list_repos "$2" "$3" "$4"
        ;;
    "get-repo")
        get_repo "$2"
        ;;
    "create-repo")
        create_repo "$2" "$3" "$4" "$5"
        ;;
    "list-issues")
        list_issues "$2" "$3" "$4"
        ;;
    "create-issue")
        create_issue "$2" "$3" "$4" "$5"
        ;;
    "update-issue")
        update_issue "$2" "$3" "$4" "$5"
        ;;
    "list-prs")
        list_prs "$2" "$3"
        ;;
    "create-pr")
        create_pr "$2" "$3" "$4" "$5" "$6"
        ;;
    "search-repos")
        search_repos "$2" "$3" "$4"
        ;;
    "search-issues")
        search_issues "$2" "$3" "$4"
        ;;
    "notifications")
        get_notifications "$2"
        ;;
    "star")
        star_repo "$2"
        ;;
    "create-gist")
        create_gist "$2" "$3" "$4" "$5"
        ;;
    *)
        echo "GitHub API Command Line Tool"
        echo ""
        echo "Usage: $0 <command> [arguments]"
        echo ""
        echo "Commands:"
        echo "  whoami                           - Show authenticated user info"
        echo "  list-repos [USER] [VIS] [LIMIT]  - List repositories"
        echo "  get-repo <OWNER/REPO>            - Get repository details"
        echo "  create-repo <NAME> [DESC] [PRIVATE] [AUTO_INIT] - Create repository"
        echo "  list-issues <OWNER/REPO> [STATE] [ASSIGNEE] - List issues"
        echo "  create-issue <OWNER/REPO> <TITLE> [BODY] [LABELS] - Create issue"
        echo "  update-issue <OWNER/REPO> <NUM> <FIELD> <VALUE> - Update issue"
        echo "  list-prs <OWNER/REPO> [STATE]    - List pull requests"
        echo "  create-pr <OWNER/REPO> <TITLE> <HEAD> [BASE] [BODY] - Create PR"
        echo "  search-repos <QUERY> [SORT] [LIMIT] - Search repositories"
        echo "  search-issues <QUERY> [SORT] [LIMIT] - Search issues/PRs"
        echo "  notifications [ALL:true/false]   - Get notifications"
        echo "  star <OWNER/REPO>                - Star a repository"
        echo "  create-gist <DESC> <FILE> [CONTENT|-] [PUBLIC] - Create gist"
        echo ""
        echo "Examples:"
        echo "  $0 whoami"
        echo "  $0 list-repos"
        echo "  $0 create-issue owner/repo \"Bug report\" \"Found a bug\""
        echo "  $0 search-repos \"language:python machine learning\""
        echo "  $0 create-pr owner/repo \"Add feature\" feature-branch"
        echo "  echo 'print(\"Hello\")' | $0 create-gist \"Python example\" hello.py -"
        echo ""
        echo "Note: Set GITHUB_TOKEN environment variable with your personal access token"
        ;;
esac