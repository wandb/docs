import os
import shutil
import re
import frontmatter

def clean_directory(directory):
    """Remove directory if it exists and create it fresh."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def get_menu_type(src_path):
    """Determine the menu type based on the source path."""
    if '/guides/' in src_path:
        return 'default'
    elif '/ref/' in src_path:
        return 'reference'
    elif '/launch/' in src_path:
        return 'launch'
    elif '/support/' in src_path:
        return 'support'
    elif '/tutorials/' in src_path:
        return 'tutorials'
    return None

def get_identifier_from_path(dest_path):
    """Generate an identifier using the full destination path after content/."""
    # Split on content/ and take everything after it
    parts = dest_path.split('content/')[1]
    
    # Remove file extension and convert path separators to hyphens
    identifier = parts.replace('.md', '').replace('/', '-')
    
    return identifier

def update_markdown_content(content, subdirectory):
    """Update relref links in markdown content."""
    def replace_relref(match):
        full_match = match.group(0)
        path_start = full_match.find('relref') + 6
        path_end = full_match.rfind('>')
        path_with_quotes = full_match[path_start:path_end].strip()
        
        # If the path includes 'content/', remove it
        clean_path = path_with_quotes.replace('content/', '')
        
        # Create the new key-value format
        clean_subdir = subdirectory.replace('content/', '')
        return f'{{{{< relref path={clean_path} lang="{clean_subdir}" >}}}}'
    
    pattern = r'\{\{<\s*relref\s*"[^"]+"\s*>\}}'
    return re.sub(pattern, replace_relref, content)

def ensure_directory_exists(filepath):
    """Create parent directories if they don't exist."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def process_markdown_file(src_path, dest_path, subdirectory):
    """Process a markdown file, updating front matter and content."""
    with open(src_path, 'r', encoding='utf-8') as file:
        post = frontmatter.load(file)
        
        # Get the menu type based on the source path
        menu_type = get_menu_type(src_path)
        
        if menu_type:
            # Generate identifier from destination path
            identifier = get_identifier_from_path(dest_path)
            
            # Initialize menu if it doesn't exist
            if 'menu' not in post:
                post['menu'] = {}
                
            # Process both the section menu (default/reference/etc) and main menu if it exists
            menus_to_process = [menu_type]
            if 'main' in post.get('menu', {}):
                menus_to_process.append('main')
            
            # Update each menu section
            for menu_name in menus_to_process:
                # Get existing menu section or create new one
                menu_section = post['menu'].get(menu_name, {})
                if not isinstance(menu_section, dict):
                    menu_section = {}
                
                # Add/update the identifier while preserving other properties
                menu_section['identifier'] = identifier
                
                # Update the menu section
                post['menu'][menu_name] = menu_section
        
        # Update content
        post.content = update_markdown_content(post.content, subdirectory)
        
        # Ensure parent directory exists before writing
        ensure_directory_exists(dest_path)
        
        # Write the modified file
        with open(dest_path, 'w', encoding='utf-8') as output:
            output.write(frontmatter.dumps(post))

def copy_and_process_directory(src, dest, subdirectory):
    """Recursively copy directory and process markdown files."""
    ensure_directory_exists(dest)
    
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dest_path = os.path.join(dest, item)
        
        if os.path.isdir(src_path):
            copy_and_process_directory(src_path, dest_path, subdirectory)
        elif src_path.endswith('.md'):
            process_markdown_file(src_path, dest_path, subdirectory)
        else:
            ensure_directory_exists(dest_path)
            shutil.copy2(src_path, dest_path)

def process_language(subdirectory):
    """Process all content for a specific language subdirectory."""
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
        # Remove 'content/' from the source path when constructing destination
        dest_path = os.path.join(subdirectory, item.replace('content/', '', 1))
        
        if os.path.isdir(src_path):
            copy_and_process_directory(src_path, dest_path, subdirectory)
        elif src_path.endswith('.md'):
            process_markdown_file(src_path, dest_path, subdirectory)

def main():
    # List of target language directories
    language_dirs = ['content/ja', 'content/ko']
    
    print("Starting i18n seed process...")
    
    # Process each language
    for lang_dir in language_dirs:
        print(f"\nProcessing {lang_dir}...")
        process_language(lang_dir)
        print(f"Completed processing {lang_dir}")
    
    print("\ni18n re-seed process complete!")

if __name__ == "__main__":
    main()