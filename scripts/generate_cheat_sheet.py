#!/usr/bin/env python3
"""
Generate SDK coding cheat sheet from ground truth code examples.

Scans MDX snippet files and creates a comprehensive cheat sheet
with a landing page and category-specific pages.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def convert_to_imperative(text: str) -> str:
    """Convert a descriptive sentence to imperative form.
    
    Examples:
        "Creates and logs a W&B artifact" -> "Create and log a W&B artifact"
        "Adds an alias" -> "Add an alias"
        "Downloads files" -> "Download files"
    """
    # Replace common present tense verbs throughout the text (not just at start)
    # This handles both first verbs and verbs in compound phrases like "creates and logs"
    verb_replacements = {
        r'\binitializes\b': 'initialize',
        r'\bInitializes\b': 'Initialize',
        r'\bcreates\b': 'create',
        r'\bCreates\b': 'Create',
        r'\badds\b': 'add',
        r'\bAdds\b': 'Add',
        r'\bdownloads\b': 'download',
        r'\bDownloads\b': 'Download',
        r'\bdeletes\b': 'delete',
        r'\bDeletes\b': 'Delete',
        r'\btracks\b': 'track',
        r'\bTracks\b': 'Track',
        r'\bupdates\b': 'update',
        r'\bUpdates\b': 'Update',
        r'\bremoves\b': 'remove',
        r'\bRemoves\b': 'Remove',
        r'\blinks\b': 'link',
        r'\bLinks\b': 'Link',
        r'\blogs\b': 'log',
        r'\bLogs\b': 'Log',
        r'\bretrieves\b': 'retrieve',
        r'\bRetrieves\b': 'Retrieve',
        r'\bfetches\b': 'fetch',
        r'\bFetches\b': 'Fetch',
        r'\bgets\b': 'get',
        r'\bGets\b': 'Get',
        r'\bsets\b': 'set',
        r'\bSets\b': 'Set',
        r'\bceates\b': 'create',  # Handle typo
        r'\bCeates\b': 'Create',  # Handle typo
    }
    
    result = text
    for pattern, replacement in verb_replacements.items():
        result = re.sub(pattern, replacement, result)
    
    return result


def remove_wandb_from_heading(text: str) -> str:
    """Remove 'W&B' from heading text since it's already mentioned in the intro.
    
    Examples:
        "Create and log a W&B artifact" -> "Create and log an artifact"
        "Initialize a W&B run" -> "Initialize a run"
        "Delete from W&B" -> "Delete"
        "Log to W&B" -> "Log"
    """
    # Remove "W&B " when followed by a word
    result = re.sub(r'\bW&B\s+', '', text)
    
    # Remove phrases like "from W&B", "to W&B", "in W&B" at the end
    result = re.sub(r'\s+(from|to|in)\s+W&B$', '', result)
    result = re.sub(r'\s+(from|to|in)\s+W&B\s+', r' ', result)
    
    # Remove "when logging it to W&B"
    result = re.sub(r'\s+when\s+logging\s+it\s+to\s+W&B', '', result)
    
    # Fix articles:
    # "a artifact/experiment/existing" -> "an artifact/experiment/existing"
    result = re.sub(r'\ba\s+(artifact|experiment|existing)', r'an \1', result)
    # "an description/registry/run" -> "a description/registry/run" (fix typos)
    result = re.sub(r'\ban\s+(description|registry|run|tag|label|table|metric)', r'a \1', result)
    
    return result


def extract_docstring_first_line(py_content: str) -> Optional[str]:
    """Extract the first sentence of the Python docstring from Python code."""
    # Match docstring (single or multi-line)
    # Single-line: """text"""
    # Multi-line: """\ntext\n..."""
    docstring_match = re.search(r'"""(.+?)"""', py_content, re.DOTALL)
    if not docstring_match:
        return None
    
    docstring = docstring_match.group(1).strip()
    
    # Get first sentence (up to period followed by space/newline, or end of string)
    # This handles both single-line and multi-line docstrings
    sentence_match = re.match(r'^(.*?\.(?:\s|$))', docstring, re.DOTALL)
    if sentence_match:
        first_sentence = sentence_match.group(1).strip()
    else:
        # No period found, use the whole docstring
        first_sentence = docstring.strip()
    
    # Clean up: remove trailing period, collapse whitespace
    first_sentence = re.sub(r'\s+', ' ', first_sentence)
    first_sentence = first_sentence.rstrip('.')
    
    # Convert to imperative form
    first_sentence = convert_to_imperative(first_sentence)
    
    return first_sentence


def infer_category_from_filename(filename: str) -> tuple[str, str]:
    """
    Infer main category and subcategory from filename.
    Returns (main_category, subcategory).
    """
    base = filename.replace('.mdx', '').lower()
    
    if base.startswith('artifact'):
        main = 'Artifacts'
        # Determine subcategory based on pattern
        if 'create' in base or 'track' in base:
            sub = 'Artifact - Creation'
        elif 'delete' in base:
            sub = 'Artifact - Deletion'
        elif 'download' in base:
            sub = 'Artifact - Downloads'
        elif 'alias' in base or 'tag' in base or 'ttl' in base:
            sub = 'Artifact - Metadata'
        elif 'update' in base:
            sub = 'Artifact - Updates'
        else:
            sub = 'Artifact - Other'
    elif base.startswith('registry'):
        main = 'Registry'
        if 'collection' in base:
            sub = 'Registry - Collections'
        else:
            sub = 'Registry - Basic Operations'
    elif base.startswith('log'):
        main = 'Logging'
        sub = 'Logging'
    elif base.startswith('run') or base.startswith('experiments'):
        main = 'Runs'
        sub = 'Run Management'
    else:
        main = 'Other'
        sub = 'Other'
    
    return main, sub


def load_tasks_from_py(py_snippets_dir: Path) -> List[Dict]:
    """Load tasks by scanning Python snippet files."""
    tasks = []
    
    for py_file in sorted(py_snippets_dir.glob('*.py')):
        # Read the Python file
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract description from docstring
        description = extract_docstring_first_line(content)
        if not description:
            print(f"   ⚠ Could not extract description from {py_file.name}")
            continue
        
        # Infer category from filename
        main_cat, sub_cat = infer_category_from_filename(py_file.name)
        
        tasks.append({
            'category': sub_cat,
            'description': description,
            'script': py_file.name
        })
    
    return tasks


def map_to_main_categories(category: str) -> str:
    """Map detailed categories to main category pages."""
    if category.startswith('Artifact'):
        return 'Artifacts'
    elif category.startswith('Registry'):
        return 'Registry'
    elif category == 'Logging':
        return 'Logging'
    elif category == 'Run Management':
        return 'Runs'
    else:
        return 'Other'


def organize_by_main_category(tasks: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """Organize tasks by main category and subcategory."""
    organized = {}
    for task in tasks:
        main_cat = map_to_main_categories(task['category'])
        sub_cat = task['category']
        
        if main_cat not in organized:
            organized[main_cat] = {}
        if sub_cat not in organized[main_cat]:
            organized[main_cat][sub_cat] = []
        
        organized[main_cat][sub_cat].append(task)
    
    return organized


def get_code_snippet_import() -> str:
    """Generate import statement for the CodeSnippet component."""
    return "import { CodeSnippet } from '/snippets/CodeSnippet.jsx';"


def generate_landing_page(main_categories: Dict[str, Dict[str, List[Dict]]]) -> str:
    """Generate the landing page that links to category pages."""
    
    # Category metadata - order matters for display
    category_info = [
        ('Runs', 'Code examples to initialize, manage, and fork W&B runs'),
        ('Logging', 'Code examples to log metrics, hyperparameters, tables, and custom data to W&B'),
        ('Artifacts', 'Code examples to create, update, download, and manage W&B Artifacts for data versioning'),
        ('Registry', 'Code examples to organize and manage model versions in W&B Registry')
    ]
    
    content = """---
title: W&B SDK Python coding cheat sheet
sidebarTitle: Overview
description: Quick reference for common W&B Python SDK patterns and code examples
---

This cheat sheet provides quick-reference Python code examples for common W&B tasks, demonstrating recommended practices. Examples use placeholder syntax like `<project>` that you can replace with your own values. Each example is a standalone snippet you can adapt to your needs.

Select a card to view code examples in that category.

<Columns cols={1}>
"""
    
    # Generate category cards in specified order
    for cat_name, cat_description in category_info:
        # Convert category name to slug
        slug = cat_name.lower()
        
        content += f'  <Card title="{cat_name}" href="/models/ref/sdk-coding-cheat-sheet/{slug}">\n'
        content += f'    {cat_description}\n'
        content += f'  </Card>\n'
    
    content += """</Columns>
"""
    
    return content


def generate_category_page(main_cat: str, subcategories: Dict[str, List[Dict]], py_snippets_dir: Path) -> str:
    """Generate a category-specific page."""
    
    category_titles = {
        'Runs': 'Runs',
        'Logging': 'Logging',
        'Artifacts': 'Artifacts',
        'Registry': 'Registry'
    }
    
    category_descriptions = {
        'Runs': 'Initialize and manage W&B runs to organize your experiments and track your work.',
        'Logging': 'Log metrics, hyperparameters, tables, and custom data to W&B.',
        'Artifacts': 'Create, update, download, and manage W&B Artifacts for data versioning.',
        'Registry': 'Work with W&B Model Registry to organize and manage model versions.',
    }
    
    title = category_titles.get(main_cat, main_cat)
    description = category_descriptions.get(main_cat, f'Code examples for {main_cat}')
    
    # Generate import for CodeSnippet component
    code_snippet_import = get_code_snippet_import()
    
    content = f"""---
title: {title}
description: {description}
---

{code_snippet_import}

{description}

"""
    
    # Add navigation back to main page
    content += "[← Back to Cheat Sheet](/models/ref/sdk-coding-cheat-sheet)\n\n"
    content += "---\n\n"
    
    # Generate flat list of all tasks across subcategories
    all_tasks = []
    for sub_cat in sorted(subcategories.keys()):
        all_tasks.extend(subcategories[sub_cat])
    
    # Add each task as an H2
    for task in all_tasks:
        # Remove W&B from headings (already mentioned in intro paragraph)
        heading = remove_wandb_from_heading(task['description'])
        content += f"## {heading}\n\n"
        
        # Check if Python file exists
        py_file = py_snippets_dir / task['script']
        if py_file.exists():
            # Use CodeSnippet component with filename
            content += f'<CodeSnippet file="{task["script"]}" />\n\n'
        else:
            # Fallback if file doesn't exist
            content += "```python\n"
            content += f"# Code example: {task['script']}\n"
            content += "# (Python snippet not found)\n"
            content += "```\n\n"
        
        content += "---\n\n"
    
    return content


def update_docs_json(docs_root: Path, main_categories: Dict[str, Dict[str, List[Dict]]]):
    """Update docs.json to include the cheat sheet structure."""
    docs_json_path = docs_root / 'docs.json'
    
    with open(docs_json_path, 'r', encoding='utf-8') as f:
        docs_json = json.load(f)
    
    # Find the Python reference section
    navigation = docs_json.get('navigation', [])
    
    # Build the new cheat sheet structure
    cheat_sheet_structure = {
        "group": "SDK Coding Cheat Sheet",
        "pages": [
            "models/ref/sdk-coding-cheat-sheet"
        ]
    }
    
    # Add category pages
    for main_cat in sorted(main_categories.keys()):
        if main_cat == 'Other':
            continue
        slug = main_cat.lower()
        cheat_sheet_structure["pages"].append(f"models/ref/sdk-coding-cheat-sheet/{slug}")
    
    # Find and update the Python reference section
    def find_and_update_python_ref(items):
        for i, item in enumerate(items):
            if isinstance(item, dict):
                # Check if this is the Python reference group
                if item.get('group') == 'Python' and 'pages' in item:
                    # Find the cheat sheet entry
                    for j, page in enumerate(item['pages']):
                        if page == 'models/ref/sdk-coding-cheat-sheet':
                            # Replace with the new structure
                            item['pages'][j] = cheat_sheet_structure
                            return True
                        elif isinstance(page, dict) and page.get('group') == 'SDK Coding Cheat Sheet':
                            # Already exists, update it
                            item['pages'][j] = cheat_sheet_structure
                            return True
                
                # Recursively search in nested pages
                if 'pages' in item:
                    if find_and_update_python_ref(item['pages']):
                        return True
        return False
    
    if find_and_update_python_ref(navigation):
        # Write back to docs.json
        with open(docs_json_path, 'w', encoding='utf-8') as f:
            json.dump(docs_json, f, indent=2)
        print(f"   ✓ Updated docs.json with cheat sheet structure")
    else:
        print(f"   ⚠ Could not find Python reference section in docs.json")


def main():
    """Generate the cheat sheet pages."""
    # Paths
    script_dir = Path(__file__).parent
    docs_root = script_dir.parent
    py_snippets_dir = docs_root / 'snippets' / 'en' / '_includes' / 'code-examples'
    output_dir = docs_root / 'models' / 'ref' / 'sdk-coding-cheat-sheet'
    landing_page = docs_root / 'models' / 'ref' / 'sdk-coding-cheat-sheet.mdx'
    
    # Check if Python snippets directory exists
    if not py_snippets_dir.exists():
        print(f"❌ Python snippets directory not found: {py_snippets_dir}")
        print("   Run sync_code_examples.sh first to sync examples")
        return 1
    
    print(f"📖 Generating cheat sheet from Python snippets in {py_snippets_dir}...")
    
    # Load and organize tasks
    tasks = load_tasks_from_py(py_snippets_dir)
    print(f"   Loaded {len(tasks)} tasks from Python files")
    
    main_categories = organize_by_main_category(tasks)
    print(f"   Organized into {len([c for c in main_categories.keys() if c != 'Other'])} main categories")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate landing page
    landing_content = generate_landing_page(main_categories)
    with open(landing_page, 'w', encoding='utf-8') as f:
        f.write(landing_content)
    print(f"   ✓ Generated landing page: {landing_page}")
    
    # Generate category pages
    for main_cat in sorted(main_categories.keys()):
        if main_cat == 'Other':
            continue
        
        slug = main_cat.lower()
        category_page = output_dir / f'{slug}.mdx'
        
        category_content = generate_category_page(
            main_cat,
            main_categories[main_cat],
            py_snippets_dir
        )
        
        with open(category_page, 'w', encoding='utf-8') as f:
            f.write(category_content)
        
        print(f"   ✓ Generated {main_cat} page: {category_page}")
    
    # Update docs.json
    update_docs_json(docs_root, main_categories)
    
    total_pages = len([c for c in main_categories.keys() if c != 'Other']) + 1
    print(f"\n✅ Generated {total_pages} pages (1 landing + {total_pages - 1} categories)")
    
    return 0


if __name__ == "__main__":
    exit(main())
