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

# Trash directory for moved files
TRASH_DIR="./trash"
mkdir -p "$TRASH_DIR"

# Generate a long random name for the content file in the trash directory
trash_file="$TRASH_DIR/$(date +%s)-$(uuidgen).md"

# Step 1: Move the content file to the trash with a unique name
echo "Moving '$content_file' to the trash: '$trash_file'..."
mv "$content_file" "$trash_file"

# Step 2: Move the file from docs to the canonical location in content, preserving history
echo "Moving '$docs_file' to '$content_file' with git mv to preserve history..."
git mv "$docs_file" "$content_file"

# Step 3: Replace the contents of the moved file with the contents from the trash file
echo "Replacing contents of '$content_file' with the original contents from '$trash_file'..."
cat "$trash_file" > "$content_file"

# Step 4: Delete the temporary file
echo "Deleting temporary file '$trash_file'..."
rm "$trash_file"

# Step 5: Stage and commit the changes
echo "Staging and committing changes in Git..."
git add "$content_file"
git commit -m "Move '$docs_file' to '$content_file' with Git history and restore original contents."

echo "File successfully updated with Git history and original contents preserved."
