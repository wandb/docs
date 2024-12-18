#!/bin/bash

# Check if arguments are provided
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <content_file_path> <docs_file_path>"
    echo "Example: $0 ./content/subdir/file.md ./docs/file.md"
    exit 1
fi

# Input paths
content_file="$1"    # Path to the file in the content directory (canonical location)
docs_file="$2"       # Path to the corresponding file in the docs directory

# Ensure both files exist
if [[ ! -f "$content_file" ]]; then
    echo "Error: Content file '$content_file' does not exist."
    exit 1
fi

if [[ ! -f "$docs_file" ]]; then
    echo "Error: Docs file '$docs_file' does not exist."
    exit 1
fi

echo "Copying contents into '$docs_file' and deleting '$content_file'"
cat "$content_file" > "$docs_file"
rm "$content_file"
git add .
git commit -m "Copying contents into '$docs_file' and deleting '$content_file'"
