#!/usr/bin/env python3
"""
Generate W&B code generation benchmark dataset from Python skill files.

This script scans the .ai/skills directory for .py files, extracts their
docstrings and code, and generates a JSON benchmark dataset following
the wandb_code_benchmark_schema.json specification.

Usage:
    python generate_benchmark.py
    python generate_benchmark.py --output custom_output.json
    python generate_benchmark.py --validate
"""

import json
import os
import ast
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Category mapping based on filename patterns
CATEGORY_PATTERNS = {
    'runs': r'^run_',
    'artifacts': r'^artifact',
    'registry': r'^registry',
    'logging': r'^log_',
    'experiments': r'^experiments',
    'tables': r'^.*table',
    'sweeps': r'^sweep',
    'models': r'^model',
}

# Key concepts to detect in code
KEY_CONCEPT_PATTERNS = {
    'wandb.init': r'wandb\.init\(',
    'context manager': r'with\s+wandb\.init',
    'run.log': r'run\.log\(',
    'wandb.Artifact': r'wandb\.Artifact\(',
    'wandb.Api': r'wandb\.Api\(',
    'artifact.add_file': r'artifact\.add_file\(',
    'artifact.add_dir': r'artifact\.add_dir\(',
    'artifact.add_reference': r'artifact\.add_reference\(',
    'run.log_artifact': r'run\.log_artifact\(',
    'artifact.save': r'artifact\.save\(\)',
    'wandb.Table': r'wandb\.Table\(',
    'run.finish': r'run\.finish\(\)',
    'wandb.config': r'wandb\.config',
    'api.artifact': r'api\.artifact\(',
    'artifact metadata': r'artifact\.metadata',
    'artifact aliases': r'artifact\.aliases',
}


def load_json_file(filepath: Path) -> Dict:
    """Load and parse JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_docstring_and_code(filepath: Path) -> Tuple[Optional[str], str]:
    """Extract docstring and code from a Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
    except SyntaxError:
        print(f"  ⚠ Warning: Syntax error in {filepath.name}, treating as plain text")
        docstring = None
    
    lines = content.split('\n')
    
    # Find where docstring ends
    if docstring:
        in_docstring = False
        docstring_end_line = 0
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                if not in_docstring:
                    in_docstring = True
                else:
                    docstring_end_line = i + 1
                    break
        
        code = '\n'.join(lines[docstring_end_line:]).strip()
    else:
        # No docstring, treat entire file as code
        code = content.strip()
    
    return docstring, code


def categorize_example(filename: str) -> str:
    """Determine category based on filename patterns."""
    for category, pattern in CATEGORY_PATTERNS.items():
        if re.match(pattern, filename):
            return category
    return 'other'


def determine_difficulty(filename: str, code: str) -> str:
    """Heuristic for difficulty based on code complexity."""
    # Count non-comment, non-blank lines
    lines = [l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
    line_count = len(lines)
    
    # Check for advanced patterns
    has_api = 'wandb.Api()' in code
    is_update = 'existing' in filename or 'update' in filename
    has_multiple_operations = code.count('artifact.') > 3 or code.count('run.') > 3
    
    if has_api or is_update or has_multiple_operations:
        return 'advanced'
    elif line_count > 15 or 'Artifact' in code or 'Table' in code:
        return 'intermediate'
    else:
        return 'basic'


def extract_placeholders(code: str) -> Dict[str, str]:
    """Extract placeholders and generate descriptions."""
    placeholders = {}
    pattern = r'<([^>]+)>'
    matches = re.findall(pattern, code)
    
    for match in set(matches):
        placeholder_full = f'<{match}>'
        # Generate human-readable description
        desc = match.replace('_', ' ').replace('/', ' or ').title()
        placeholders[placeholder_full] = desc
    
    return placeholders


def extract_key_concepts(code: str) -> List[str]:
    """Extract key W&B concepts from code using pattern matching."""
    concepts = []
    
    for concept, pattern in KEY_CONCEPT_PATTERNS.items():
        if re.search(pattern, code):
            concepts.append(concept)
    
    return concepts


def infer_common_mistakes(filename: str, code: str, concepts: List[str]) -> List[Dict[str, str]]:
    """Infer common mistakes based on code patterns."""
    mistakes = []
    
    # Check for context manager usage
    if 'wandb.init' in code and 'with wandb.init' not in code:
        mistakes.append({
            "mistake": "Should use context manager (with statement) for wandb.init"
        })
    
    # Check for artifact save
    if 'wandb.Api' in code and 'artifact' in code.lower() and 'save()' in code:
        mistakes.append({
            "mistake": "Forgetting to call artifact.save() to persist changes"
        })
    
    # Check for proper imports
    if 'wandb.' in code and 'import wandb' not in code:
        mistakes.append({
            "mistake": "Missing 'import wandb' statement"
        })
    
    return mistakes


def process_skill_file(filepath: Path, verbose: bool = False) -> Optional[Dict]:
    """Process a single skill file and return example dict."""
    filename = filepath.stem
    
    if verbose:
        print(f"  Processing {filepath.name}...")
    
    try:
        docstring, code = extract_docstring_and_code(filepath)
        
        if not docstring:
            print(f"  ⚠ Warning: No docstring in {filepath.name}, skipping")
            return None
        
        category = categorize_example(filename)
        difficulty = determine_difficulty(filename, code)
        key_concepts = extract_key_concepts(code)
        placeholders = extract_placeholders(code)
        common_mistakes = infer_common_mistakes(filename, code, key_concepts)
        
        example = {
            "id": filename,
            "category": category,
            "difficulty": difficulty,
            "prompt": docstring.strip(),
            "reference_code": code,
            "key_concepts": key_concepts,
            "source_file": filepath.name
        }
        
        # Add optional fields if they exist
        if common_mistakes:
            example["common_mistakes"] = common_mistakes
        if placeholders:
            example["placeholders"] = placeholders
        
        return example
        
    except Exception as e:
        print(f"  ✗ Error processing {filepath.name}: {e}")
        return None


def generate_benchmark(
    skills_dir: Path,
    output_file: Path,
    verbose: bool = False
) -> Dict:
    """Generate complete benchmark from skill files."""
    
    if verbose:
        print(f"Scanning {skills_dir} for Python files...")
    
    # Files to exclude from benchmark (utility scripts)
    exclude_files = {
        'generate_benchmark.py',
        'validate_benchmark.py',
        'add_example.py'
    }
    
    examples = []
    py_files = sorted(skills_dir.glob('*.py'))
    
    for py_file in py_files:
        # Skip utility scripts
        if py_file.name in exclude_files:
            if verbose:
                print(f"  Skipping utility script {py_file.name}")
            continue
        example = process_skill_file(py_file, verbose=verbose)
        if example:
            examples.append(example)
            if verbose:
                print(f"    ✓ Added {example['id']}")
    
    # Create metadata
    metadata = {
        "version": "1.0.0",
        "created_date": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "description": "W&B SDK code generation evaluation benchmark for teaching core concepts",
        "sdk_version": ">=0.18.0",
        "num_examples": len(examples)
    }
    
    benchmark = {
        "metadata": metadata,
        "examples": examples
    }
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n✓ Generated benchmark with {len(examples)} examples")
        print(f"✓ Saved to {output_file}")
        
        # Print summary statistics
        print_summary(examples)
    
    return benchmark


def print_summary(examples: List[Dict]):
    """Print summary statistics about the dataset."""
    categories = {}
    difficulties = {}
    
    for ex in examples:
        cat = ex['category']
        diff = ex['difficulty']
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"\nDataset Summary:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Categories:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    print(f"  Difficulties:")
    for diff, count in sorted(difficulties.items()):
        print(f"    {diff}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate W&B code generation benchmark dataset"
    )
    parser.add_argument(
        '--skills-dir',
        type=Path,
        default=Path(__file__).parent,
        help='Directory containing skill .py files (default: script directory)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent / 'wandb_code_benchmark.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate output against schema after generation'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Generate benchmark
    benchmark = generate_benchmark(
        skills_dir=args.skills_dir,
        output_file=args.output,
        verbose=args.verbose
    )
    
    # Validate if requested
    if args.validate:
        print("\nValidating generated benchmark...")
        from validate_benchmark import validate_benchmark_file
        is_valid = validate_benchmark_file(args.output, verbose=args.verbose)
        exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
