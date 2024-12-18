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

# Check if the source file exists
if [[ ! -f "$docs_file" ]]; then
    echo "Error: Docs file '$docs_file' does not exist."
    exit 1
fi

# Ensure the destination directory exists
content_dir=$(dirname "$content_file")
if [[ ! -d "$content_dir" ]]; then
    echo "Creating destination directory: $content_dir"
    mkdir -p "$content_dir"
fi

# Move the file with git mv
echo "Moving $docs_file to $content_file"
git mv "$docs_file" "$content_file"
