import os
import shutil
import sys
import re
import frontmatter

def clean_directory(directory):
    """Remove directory if it exists and create it fresh."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def update_markdown_content(content, subdirectory):
    """Update relref links in markdown content."""
    def replace_relref(match):
        # Get the entire matched relref
        full_match = match.group(0)
        # Get the path part including quotes
        path_start = full_match.find('relref') + 6  # skip 'relref' and space
        path_end = full_match.rfind('>')
        path_with_quotes = full_match[path_start:path_end].strip()
        
        # If the path includes 'content/', remove it
        clean_path = path_with_quotes.replace('content/', '')
        
        # Create the new key-value format
        clean_subdir = subdirectory.replace('content/', '')
        return f'{{{{< relref path={clean_path} lang="{clean_subdir}" >}}}}'
    
    # Find all relref shortcodes (with any path format)
    pattern = r'\{\{<\s*relref\s*"[^"]+"\s*>\}}'
    return re.sub(pattern, replace_relref, content)

def process_markdown_file(src_path, dest_path, subdirectory):
    """Process a markdown file, updating front matter and content."""
    with open(src_path, 'r', encoding='utf-8') as file:
        post = frontmatter.load(file)
        
        # Get the clean subdirectory (without content/)
        clean_subdir = subdirectory.replace('content/', '')
        
        # Handle nested menu structure
        if 'menu' in post:
            for menu_section in post['menu'].values():
                if isinstance(menu_section, dict):
                    # Update identifier if it exists
                    if 'identifier' in menu_section:
                        menu_section['identifier'] = f"{clean_subdir}-{menu_section['identifier']}"
                    
                    # Update parent if it exists
                    if 'parent' in menu_section:
                        menu_section['parent'] = f"{clean_subdir}-{menu_section['parent']}"
        
        # Update content
        post.content = update_markdown_content(post.content, subdirectory)
        
        # Write the modified file
        with open(dest_path, 'w', encoding='utf-8') as output:
            output.write(frontmatter.dumps(post))

def copy_and_process_directory(src, dest, subdirectory):
    """Recursively copy directory and process markdown files."""
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dest_path = os.path.join(dest, item)
        
        if os.path.isdir(src_path):
            copy_and_process_directory(src_path, dest_path, subdirectory)
        elif src_path.endswith('.md'):
            process_markdown_file(src_path, dest_path, subdirectory)
        else:
            shutil.copy2(src_path, dest_path)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <subdirectory>")
        print("Example: python script.py content/ja")
        sys.exit(1)
    
    subdirectory = sys.argv[1]
    
    # List of directories and files to copy
    items_to_copy = [
        'content/guides',
        'content/launch',
        'content/ref',
        'content/support',
        'content/tutorials',
        'content/_index.md'
    ]
    
    # Clean and create target directory
    clean_directory(subdirectory)
    
    # Process each item in the list
    for item in items_to_copy:
        src_path = item
        # Remove 'content/' from the destination path and add it to target_dir
        dest_path = os.path.join(subdirectory, item.replace('content/', '', 1))
        
        if os.path.isdir(src_path):
            dest_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            copy_and_process_directory(src_path, dest_path, subdirectory)
        elif src_path.endswith('.md'):
            dest_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            process_markdown_file(src_path, dest_path, subdirectory)

if __name__ == "__main__":
    main()