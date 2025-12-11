# W&B Code Generation Evaluation Benchmark

This directory contains a benchmark dataset for evaluating an LLM model's capability to generate high-quality W&B SDK code. The benchmark is designed to help select and evaluate models for generating code examples to help teach W&B usage.

## Overview

The benchmark consists of:
- **Python skill files** (`.py`): Individual code examples with docstring prompts
- **JSON benchmark dataset**: Structured dataset generated from skill files
- **JSON Schema**: Formal specification for validation
- **Tooling**: Scripts for generation, validation, and management

## Directory Structure

```
.ai/skills/
├── README.md                              # This file
├── wandb_code_benchmark_schema.json       # JSON Schema definition
├── wandb_code_benchmark.json              # Generated benchmark dataset
├── generate_benchmark.py                  # Generate dataset from .py files
├── validate_benchmark.py                  # Validate JSON against schema
├── add_example.py                         # Add/update single examples
├── SKILLS.md                              # Overview doc for skill files
├── run_init.py                            # Example: Initialize a run
├── artifact_create.py                     # Example: Create an artifact
├── log_metric.py                          # Example: Log metrics
└── ...                                    # More skill files
```

## Skill File Format

Each skill file is a Python file with:

1. **Docstring** (required): Natural language prompt describing what to generate
2. **Code** (required): Canonical implementation following W&B best practices
3. **Placeholders**: Use angle brackets for values (e.g., `<project>`, `<artifact_name>`)

### Example Skill File

```python
"""
Initializes a W&B run and logs a metric.

This demonstrates the basic pattern for tracking experiments.
"""
import wandb

with wandb.init(project="<project>") as run:
    # Training code here
    run.log({"accuracy": 0.95})
```

### Naming Convention

Use descriptive snake_case names that indicate the action:
- `run_init.py` - Initialize a run
- `artifact_create.py` - Create an artifact
- `registry_link_artifact_existing.py` - Link existing artifact to registry

## Usage

### Generate Full Benchmark

Generate the complete JSON dataset from all `.py` files:

```bash
# Basic generation
python generate_benchmark.py

# With validation
python generate_benchmark.py --validate

# Custom output location
python generate_benchmark.py --output custom_path.json

# Verbose output
python generate_benchmark.py -v
```

### Validate Benchmark

Validate an existing benchmark file:

```bash
# Validate default file
python validate_benchmark.py wandb_code_benchmark.json

# Verbose validation
python validate_benchmark.py wandb_code_benchmark.json -v

# Custom schema
python validate_benchmark.py --schema custom_schema.json benchmark.json
```

### Add Single Example

Add a new example or update an existing one:

```bash
# Add new example
python add_example.py new_skill.py

# Update existing example
python add_example.py --update existing_skill.py

# Skip validation
python add_example.py --no-validate new_skill.py
```

## Workflow

### Adding a New Example

1. Create a new `.py` file following the skill file format
2. Add the example to the benchmark:
   ```bash
   python add_example.py my_new_skill.py -v
   ```
3. Commit both the `.py` file and updated `wandb_code_benchmark.json`

### Updating an Example

1. Edit the `.py` file
2. Update the benchmark:
   ```bash
   python add_example.py --update my_skill.py -v
   ```
3. Commit changes

### Regenerating Full Benchmark

Regenerate from scratch (useful after major changes):

```bash
python generate_benchmark.py --validate -v
```

## Using the Benchmark

### With Weave

```python
import weave
import json

# Load benchmark
with open('.ai/skills/wandb_code_benchmark.json') as f:
    benchmark = json.load(f)

# Use examples as dataset
dataset = benchmark['examples']

# Create evaluation
evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[your_scorers]
)

# Evaluate model
weave.init('your-project')
await evaluation.evaluate(your_model)
```

### Filtering Examples

```python
# Filter by category
artifact_examples = [
    ex for ex in benchmark['examples'] 
    if ex['category'] == 'artifacts'
]

# Filter by difficulty
basic_examples = [
    ex for ex in benchmark['examples']
    if ex['difficulty'] == 'basic'
]

# Filter by concept
init_examples = [
    ex for ex in benchmark['examples']
    if 'wandb.init' in ex['key_concepts']
]
```

## Schema Specification

The benchmark follows a strict JSON Schema (`wandb_code_benchmark_schema.json`). Key requirements:

- **Unique IDs**: Each example must have a unique, snake_case ID
- **Categories**: Must be one of: runs, artifacts, registry, logging, experiments, tables, sweeps, models, other
- **Difficulties**: Must be: basic, intermediate, or advanced
- **Required fields**: id, category, difficulty, prompt, reference_code, key_concepts
- **Optional fields**: common_mistakes, placeholders, source_file, tags

## Best Practices

### Writing Skill Files

1. **Clear prompts**: Write docstrings as if instructing a developer
2. **Complete examples**: Include all necessary imports and context
3. **Use placeholders**: Use `<placeholder>` format for variable values
4. **Follow patterns**: Match the style of existing examples
5. **Add comments**: Explain non-obvious code

### Maintaining Quality

1. **Validate regularly**: Run validation after changes
2. **Test code**: Ensure reference code actually works
3. **Keep updated**: Update examples when SDK changes
4. **Document mistakes**: Add common_mistakes to help scorers

## Dependencies

- Python 3.7+
- `jsonschema` (for validation): `pip install jsonschema`

## CI/CD Integration

Consider adding to your workflow:

```yaml
# .github/workflows/validate-benchmark.yml
name: Validate Benchmark
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install jsonschema
      - run: python .ai/skills/validate_benchmark.py .ai/skills/wandb_code_benchmark.json
```

## Contributing

When adding new examples:
1. Follow the skill file format
2. Use meaningful IDs and categories
3. Include comprehensive prompts
4. Test your code
5. Run validation before committing
6. Update this README if adding new categories

## Questions?

Contact the W&B docs team or open an issue in the docs repository.
