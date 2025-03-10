import os
import yaml
import re
import json

def preprocess_js_to_json(js_content):
    """
    Preprocess JavaScript object notation into JSON-compatible format.
    """
    js_content = re.sub(r"//.*", "", js_content)  # Remove single-line comments
    js_content = re.sub(r"/\*.*?\*/", "", js_content, flags=re.DOTALL)  # Remove multi-line comments
    js_content = re.sub(r"export\s+default\s+", "", js_content)  # Remove the `export default` line
    js_content = re.sub(r"'", '"', js_content)  # Replace single quotes with double quotes
    js_content = re.sub(r"(\s|{|,)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', js_content)  # Quote unquoted keys
    js_content = re.sub(r",(\s*[\]}])", r"\1", js_content)  # Remove trailing commas
    return js_content.strip(";").strip()

def slugify_label(label):
    """
    Convert a label to an ASCII slug.
    """
    return re.sub(r"[^a-zA-Z0-9]+", "-", label.lower()).strip("-")

def read_existing_file_content(file_path):
    """
    Read the content and front matter from an existing file.
    """
    if not os.path.exists(file_path):
        return {}, ""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Split front matter and body
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) > 2:
                front_matter = yaml.safe_load(parts[1])
                body = parts[2].strip()
                return front_matter, body
    return {}, content

def create_markdown_file(path, front_matter, content=""):
    """
    Create a Markdown file with front matter.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml_front_matter = yaml.dump(front_matter, default_flow_style=False)
        f.write(f"---\n{yaml_front_matter}---\n\n{content}")

def process_items(items, top_menu, base_path="content", docs_path="docs", parent_slug=None):
    """
    Process menu items recursively and generate markdown files.
    """
    for item in items:
        if isinstance(item, str):
            # Extract the path and filename
            parts = item.split("/")
            file_path = os.path.join(base_path, *parts) + ".md"
            docs_file_path = os.path.join(docs_path, *parts) + ".md"
            identifier = parts[-1]  # Use only the filename without extension as the identifier
            
            # Read the old content and front matter
            old_front_matter, old_content = read_existing_file_content(docs_file_path)
            
            # Prepare new front matter
            front_matter = {
                "menu": {
                    top_menu: {
                        "identifier": identifier,
                        "parent": parent_slug
                    }
                }
            }
            # Add extra keys from old front matter
            for key in ["title", "description", "tags"]:
                if key in old_front_matter:
                    front_matter[key] = old_front_matter[key]
            
            # Create the new file
            create_markdown_file(file_path, front_matter, old_content)
        elif isinstance(item, dict):
            if item.get("type") == "category":
                category_label = item["label"]
                category_slug = slugify_label(category_label)
                # Process category items
                if "items" in item:
                    process_items(item["items"], top_menu, base_path, docs_path, parent_slug=category_slug)
                if "link" in item and "id" in item["link"]:
                    # Handle linked categories
                    parts = item["link"]["id"].split("/")
                    file_path = os.path.join(base_path, *parts) + ".md"
                    docs_file_path = os.path.join(docs_path, *parts) + ".md"
                    identifier = parts[-1]  # Use only the filename without extension as the identifier
                    
                    # Read the old content and front matter
                    old_front_matter, old_content = read_existing_file_content(docs_file_path)
                    
                    # Prepare new front matter
                    front_matter = {
                        "menu": {
                            top_menu: {
                                "identifier": identifier,
                                "parent": parent_slug
                            }
                        }
                    }
                    # Add extra keys from old front matter
                    for key in ["title", "description", "tags"]:
                        if key in old_front_matter:
                            front_matter[key] = old_front_matter[key]
                    
                    # Create the new file
                    create_markdown_file(file_path, front_matter, old_content)
                elif not "link" in item:
                    # Create a placeholder front matter if the category has no associated link
                    placeholder_path = os.path.join(base_path, category_slug, "_index.md")
                    front_matter = {
                        "menu": {
                            top_menu: {
                                "identifier": category_slug,
                                "parent": parent_slug
                            }
                        }
                    }
                    create_markdown_file(placeholder_path, front_matter)

# Load and preprocess the sidebar.js file
with open("sidebars.js", "r", encoding="utf-8") as f:
    js_data = f.read()

# Preprocess the JavaScript content
json_data = preprocess_js_to_json(js_data)

# Safely parse the JSON
try:
    menu_data = json.loads(json_data)
except json.JSONDecodeError as e:
    print("Error decoding JSON:", e)
    print("Preprocessed JSON data:\n", json_data)
    exit(1)

# Generate markdown files for each top-level menu
for menu_name, items in menu_data.items():
    process_items(items, menu_name.lower())

print("Markdown files generated successfully.")
