#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

# Directories to process
directories = [
    'content/support/kb-articles',
    'content/ja/support/kb-articles',
    'content/ko/support/kb-articles'
]

def extract_front_matter(content):
    """Extract front matter from markdown content."""
    front_matter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
    match = front_matter_pattern.match(content)
    
    if match:
        front_matter = match.group(1)
        remaining_content = content[match.end():]
        return front_matter, remaining_content
    else:
        return "", content

def add_translation_key(file_path):
    """Add translationKey to the front matter of a markdown file."""
    file_name = os.path.basename(file_path)
    translation_key = os.path.splitext(file_name)[0]
    
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract front matter
        front_matter, remaining_content = extract_front_matter(content)
        
        # Check if translationKey already exists
        if re.search(r'^translationKey:', front_matter, re.MULTILINE):
            print(f"File {file_path} already has a translationKey")
            return
        
        # Add translationKey to front matter
        if front_matter:
            updated_front_matter = front_matter + f"\ntranslationKey: {translation_key}"
        else:
            updated_front_matter = f"translationKey: {translation_key}"
        
        # Combine updated front matter with remaining content
        updated_content = f"---\n{updated_front_matter}\n---\n{remaining_content}"
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        
        print(f"Added translationKey: {translation_key} to {file_path}")
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")

def process_directory(directory):
    """Process all markdown files in a directory."""
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            print(f"Directory {directory} does not exist.")
            return
        
        # Find all markdown files in the directory
        markdown_files = list(directory_path.glob('**/*.md'))
        
        if not markdown_files:
            print(f"No markdown files found in {directory}")
            return
        
        print(f"Processing {len(markdown_files)} files in {directory}...")
        
        # Process each markdown file
        for file_path in markdown_files:
            add_translation_key(file_path)
            
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")

def main():
    """Main function to process all directories."""
    print("Starting to add translationKey to markdown files...")
    
    for directory in directories:
        process_directory(directory)
    
    print("Process completed.")

if __name__ == "__main__":
    main()
