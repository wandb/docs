#!/usr/bin/env python3
"""
Add or update a single example in the W&B code generation benchmark.

Usage:
    # Add from a new Python file
    python add_example.py new_skill.py
    
    # Update existing example
    python add_example.py --update existing_skill.py
    
    # Specify benchmark file
    python add_example.py --benchmark custom_benchmark.json new_skill.py
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

# Import functions from other scripts
from generate_benchmark import process_skill_file, load_json_file
from validate_benchmark import validate_benchmark_file


def add_or_update_example(
    skill_file: Path,
    benchmark_file: Path,
    update: bool = False,
    validate: bool = True,
    verbose: bool = False
) -> bool:
    """Add or update a single example in the benchmark."""
    
    # Process the skill file
    if verbose:
        print(f"Processing {skill_file}...")
    
    new_example = process_skill_file(skill_file, verbose=verbose)
    
    if not new_example:
        print(f"❌ Failed to process {skill_file}")
        return False
    
    # Load existing benchmark
    if benchmark_file.exists():
        benchmark = load_json_file(benchmark_file)
    else:
        # Create new benchmark
        if verbose:
            print(f"Creating new benchmark file {benchmark_file}")
        benchmark = {
            "metadata": {
                "version": "1.0.0",
                "created_date": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "description": "W&B SDK code generation evaluation benchmark",
                "sdk_version": ">=0.18.0",
                "num_examples": 0
            },
            "examples": []
        }
    
    # Check if example already exists
    existing_idx = None
    for i, ex in enumerate(benchmark['examples']):
        if ex['id'] == new_example['id']:
            existing_idx = i
            break
    
    if existing_idx is not None:
        if update:
            if verbose:
                print(f"Updating existing example '{new_example['id']}'")
            benchmark['examples'][existing_idx] = new_example
        else:
            print(f"❌ Example '{new_example['id']}' already exists")
            print(f"   Use --update to replace it")
            return False
    else:
        if verbose:
            print(f"Adding new example '{new_example['id']}'")
        benchmark['examples'].append(new_example)
    
    # Update metadata
    benchmark['metadata']['last_updated'] = datetime.now().isoformat()
    benchmark['metadata']['num_examples'] = len(benchmark['examples'])
    
    # Save benchmark
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    
    action = 'updated' if existing_idx is not None else 'added'
    print(f"✓ Successfully {action} example '{new_example['id']}'")
    print(f"✓ Saved to {benchmark_file}")
    
    # Validate if requested
    if validate:
        if verbose:
            print("\nValidating benchmark...")
        is_valid = validate_benchmark_file(benchmark_file, verbose=verbose)
        if not is_valid:
            print("⚠ Warning: Validation failed, but changes were saved")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add or update a single example in the benchmark"
    )
    parser.add_argument(
        'skill_file',
        type=Path,
        help='Python skill file to add'
    )
    parser.add_argument(
        '--benchmark',
        type=Path,
        default=Path(__file__).parent / 'wandb_code_benchmark.json',
        help='Benchmark JSON file to update'
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update existing example if it exists'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation after adding'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    success = add_or_update_example(
        skill_file=args.skill_file,
        benchmark_file=args.benchmark,
        update=args.update,
        validate=not args.no_validate,
        verbose=args.verbose
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
