#!/usr/bin/env python3
"""
Generate SDK coding cheat sheet from ground truth code examples.

Reads the llm_evaluation_tasks.csv and creates a comprehensive cheat sheet
with a landing page and category-specific pages.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List


def load_tasks(csv_path: Path) -> List[Dict]:
    """Load tasks from CSV file."""
    tasks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Reference Script'):
                tasks.append({
                    'category': row['Category'].strip(),
                    'description': row['Task Description'].strip(),
                    'difficulty': row['Difficulty'].strip(),
                    'script': row['Reference Script'].strip()
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


def get_all_imports(tasks: List[Dict]) -> str:
    """Generate import statements for all snippets."""
    all_scripts = set()
    for task in tasks:
        if task.get('script'):
            all_scripts.add(task['script'])
    
    imports = []
    for script in sorted(all_scripts):
        # Convert filename to a valid component name
        # e.g., run_init.py -> RunInit
        component_name = ''.join(word.capitalize() for word in script.replace('.py', '').split('_'))
        imports.append(f"import {component_name} from '/snippets/en/_includes/code-examples/{script.replace('.py', '.mdx')}';")
    
    return '\n'.join(imports)


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


def generate_category_page(main_cat: str, subcategories: Dict[str, List[Dict]], all_tasks: List[Dict], mdx_snippets_dir: Path) -> str:
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
    
    # Generate imports
    imports = get_all_imports(all_tasks)
    
    content = f"""---
title: {title}
description: {description}
---

{imports}

{description}

"""
    
    # Add navigation back to main page
    content += "[← Back to Cheat Sheet](/models/ref/sdk-coding-cheat-sheet)\n\n"
    content += "---\n\n"
    
    # Generate table of contents for subcategories
    if len(subcategories) > 1:
        content += "## Quick Navigation\n\n"
        for sub_cat in sorted(subcategories.keys()):
            anchor = sub_cat.lower().replace(' ', '-').replace('--', '-').replace('--', '-')
            content += f"- [{sub_cat}](#{anchor})\n"
        content += "\n---\n\n"
    
    # Generate sections for each subcategory
    for sub_cat in sorted(subcategories.keys()):
        tasks = subcategories[sub_cat]
        
        # Only show subcategory header if there are multiple subcategories
        if len(subcategories) > 1:
            content += f"## {sub_cat}\n\n"
        
        # Add tasks for this subcategory
        for task in tasks:
            content += f"### {task['description']}\n\n"
            
            # Check if MDX file exists
            mdx_file = mdx_snippets_dir / task['script'].replace('.py', '.mdx')
            if mdx_file.exists():
                # Convert filename to component name
                component_name = ''.join(word.capitalize() for word in task['script'].replace('.py', '').split('_'))
                # Use MDX component syntax
                content += f"<{component_name} />\n\n"
            else:
                # Fallback if file doesn't exist
                content += "```python\n"
                content += f"# Code example: {task['script']}\n"
                content += "# (MDX snippet not found)\n"
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
    csv_path = docs_root / 'snippets' / 'code-examples' / 'llm_evaluation_tasks.csv'
    mdx_snippets_dir = docs_root / 'snippets' / 'en' / '_includes' / 'code-examples'
    output_dir = docs_root / 'models' / 'ref' / 'sdk-coding-cheat-sheet'
    landing_page = docs_root / 'models' / 'ref' / 'sdk-coding-cheat-sheet.mdx'
    
    # Check if CSV exists
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("   Run sync_code_examples.sh first to copy examples")
        return 1
    
    print(f"📖 Generating cheat sheet from {csv_path}...")
    
    # Load and organize tasks
    tasks = load_tasks(csv_path)
    print(f"   Loaded {len(tasks)} tasks")
    
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
            tasks,
            mdx_snippets_dir
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
