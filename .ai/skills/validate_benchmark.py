#!/usr/bin/env python3
"""
Validate W&B code generation benchmark JSON against schema.

Usage:
    python validate_benchmark.py wandb_code_benchmark.json
    python validate_benchmark.py --schema custom_schema.json benchmark.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

try:
    import jsonschema
    from jsonschema import validate, Draft7Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def load_json_file(filepath: Path) -> Dict:
    """Load and parse JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_benchmark_file(
    benchmark_file: Path,
    schema_file: Path = None,
    verbose: bool = False
) -> bool:
    """Validate benchmark JSON file against schema."""
    
    if not HAS_JSONSCHEMA:
        print("❌ jsonschema package not installed")
        print("   Install with: pip install jsonschema")
        return False
    
    # Load schema
    if schema_file is None:
        schema_file = Path(__file__).parent / 'wandb_code_benchmark_schema.json'
    
    if not schema_file.exists():
        print(f"❌ Schema file not found: {schema_file}")
        return False
    
    if verbose:
        print(f"Loading schema from {schema_file}")
    
    try:
        schema = load_json_file(schema_file)
    except Exception as e:
        print(f"❌ Error loading schema: {e}")
        return False
    
    # Load benchmark
    if not benchmark_file.exists():
        print(f"❌ Benchmark file not found: {benchmark_file}")
        return False
    
    if verbose:
        print(f"Loading benchmark from {benchmark_file}")
    
    try:
        benchmark = load_json_file(benchmark_file)
    except Exception as e:
        print(f"❌ Error loading benchmark: {e}")
        return False
    
    # Validate
    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(benchmark))
    
    if errors:
        print(f"\n❌ Validation failed with {len(errors)} error(s):\n")
        for i, error in enumerate(errors, 1):
            path = ' -> '.join(str(p) for p in error.path) if error.path else 'root'
            print(f"  {i}. At {path}:")
            print(f"     {error.message}")
            if verbose and error.context:
                for ctx_error in error.context:
                    print(f"       - {ctx_error.message}")
        return False
    
    # Additional custom validations
    custom_errors = validate_custom_rules(benchmark, verbose)
    
    if custom_errors:
        print(f"\n⚠ Custom validation warnings ({len(custom_errors)}):\n")
        for i, error in enumerate(custom_errors, 1):
            print(f"  {i}. {error}")
        # Don't fail on custom warnings
    
    print(f"\n✓ Validation successful!")
    if verbose:
        print(f"  - {benchmark['metadata']['num_examples']} examples")
        print(f"  - Version: {benchmark['metadata']['version']}")
    
    return True


def validate_custom_rules(benchmark: Dict, verbose: bool = False) -> List[str]:
    """Additional custom validation rules beyond schema."""
    warnings = []
    
    # Check for duplicate IDs
    ids = [ex['id'] for ex in benchmark['examples']]
    duplicates = [id for id in ids if ids.count(id) > 1]
    if duplicates:
        warnings.append(f"Duplicate example IDs found: {set(duplicates)}")
    
    # Check that each example has 'import wandb' in reference_code
    for ex in benchmark['examples']:
        if 'import wandb' not in ex['reference_code']:
            warnings.append(
                f"Example '{ex['id']}' missing 'import wandb' in reference_code"
            )
    
    # Check that prompts are sufficiently detailed
    for ex in benchmark['examples']:
        if len(ex['prompt']) < 20:
            warnings.append(
                f"Example '{ex['id']}' has very short prompt ({len(ex['prompt'])} chars)"
            )
    
    # Validate placeholder consistency
    import re
    for ex in benchmark['examples']:
        # Find placeholders in code
        code_placeholders = set(re.findall(r'<[^>]+>', ex['reference_code']))
        declared_placeholders = set(ex.get('placeholders', {}).keys())
        
        undocumented = code_placeholders - declared_placeholders
        if undocumented:
            warnings.append(
                f"Example '{ex['id']}' has undocumented placeholders: {undocumented}"
            )
    
    return warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate W&B code generation benchmark JSON"
    )
    parser.add_argument(
        'benchmark_file',
        type=Path,
        help='Path to benchmark JSON file to validate'
    )
    parser.add_argument(
        '--schema',
        type=Path,
        help='Path to schema file (default: wandb_code_benchmark_schema.json)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    is_valid = validate_benchmark_file(
        args.benchmark_file,
        args.schema,
        args.verbose
    )
    
    exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
