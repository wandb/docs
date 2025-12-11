# Quick Start Guide

## What You Have

You now have a complete W&B code generation evaluation benchmark system with:

- ✅ **30 skill examples** covering runs, artifacts, logging, and registry operations
- ✅ **JSON Schema** for validation (`wandb_code_benchmark_schema.json`)
- ✅ **Generated benchmark** (`wandb_code_benchmark.json`)
- ✅ **Three utility scripts** for managing the benchmark
- ✅ **Complete documentation** (README.md)

## Quick Commands

```bash
# In .ai/skills/ directory

# Generate benchmark from all .py files
python3 generate_benchmark.py -v

# Validate the benchmark
python3 validate_benchmark.py wandb_code_benchmark.json -v

# Add a new example
python3 add_example.py new_skill.py -v

# Update an existing example
python3 add_example.py --update existing_skill.py -v
```

## Using with Weave for Model Evaluation

Here's a complete example of how to use this benchmark to evaluate LLMs:

```python
import weave
import json
import asyncio
from openai import OpenAI

# 1. Load the benchmark
with open('.ai/skills/wandb_code_benchmark.json', 'r') as f:
    benchmark = json.load(f)

dataset = benchmark['examples']

# 2. Create a model to evaluate
class CodeGenerationModel(weave.Model):
    model_name: str
    system_prompt: str = "You are an expert in W&B Python SDK. Generate clean, idiomatic code following best practices."
    
    @weave.op()
    def predict(self, prompt: str) -> dict:
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Write Python code that: {prompt}"}
            ],
            temperature=0.2
        )
        return {"generated_code": response.choices[0].message.content}

# 3. Define scoring functions
@weave.op()
def has_import_scorer(output: dict) -> dict:
    """Check if code imports wandb."""
    generated = output.get('generated_code', '')
    return {"has_wandb_import": 'import wandb' in generated}

@weave.op()
def contains_key_concepts_scorer(key_concepts: list, output: dict) -> dict:
    """Check if generated code contains key W&B concepts."""
    generated = output.get('generated_code', '')
    matches = sum(1 for concept in key_concepts if concept in generated)
    score = matches / len(key_concepts) if key_concepts else 0
    return {
        "key_concepts_coverage": score,
        "concepts_found": matches,
        "concepts_total": len(key_concepts)
    }

@weave.op()
def uses_context_manager_scorer(output: dict) -> dict:
    """Check if code uses context manager for wandb.init."""
    generated = output.get('generated_code', '')
    has_context_manager = 'with wandb.init' in generated
    return {"uses_context_manager": has_context_manager}

# 4. Create evaluation
evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[has_import_scorer, contains_key_concepts_scorer, uses_context_manager_scorer]
)

# 5. Run evaluation
weave.init('wandb-code-eval')

async def evaluate_models():
    # Evaluate multiple models
    models_to_test = [
        CodeGenerationModel(model_name="gpt-4o-mini"),
        CodeGenerationModel(model_name="gpt-4o"),
    ]
    
    for model in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating {model.model_name}")
        print(f"{'='*60}\n")
        
        results = await evaluation.evaluate(model)
        print(f"\nResults: {results}")

# Run it
asyncio.run(evaluate_models())
```

## Filtering Examples for Specific Tests

```python
# Test only basic examples
basic_dataset = [ex for ex in dataset if ex['difficulty'] == 'basic']

# Test only artifact operations
artifact_dataset = [ex for ex in dataset if ex['category'] == 'artifacts']

# Test only examples with specific concepts
init_dataset = [ex for ex in dataset if 'wandb.init' in ex['key_concepts']]

# Create focused evaluation
basic_eval = weave.Evaluation(
    dataset=basic_dataset,
    scorers=[your_scorers]
)
```

## Adding New Examples

1. **Create a skill file** (e.g., `sweep_create.py`):

```python
"""
Create and run a W&B sweep for hyperparameter optimization.

This example shows how to define a sweep configuration and start a sweep.
"""
import wandb

# Define sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.1},
        'batch_size': {'values': [16, 32, 64]}
    }
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="<project>")

# Run sweep agent
wandb.agent(sweep_id, function=train_function, count=10)
```

2. **Add it to the benchmark**:

```bash
python3 add_example.py sweep_create.py -v
```

3. **Commit both files**:

```bash
git add sweep_create.py wandb_code_benchmark.json
git commit -m "Add sweep creation example"
```

## Next Steps

1. **Select models to evaluate** - Consider GPT-4o-mini, Claude 3.5 Haiku, Deepseek Coder
2. **Run initial evaluation** - Use the Weave example above
3. **Analyze results** - Look at concept coverage, context manager usage, etc.
4. **Iterate** - Add more examples for concepts that models struggle with
5. **Share findings** - Document which models work best for W&B code generation

## Tips

- Start with **basic difficulty** examples to establish baseline
- Focus on **key_concepts coverage** as a primary metric
- Check for **context manager usage** (best practice for wandb.init)
- Look for **placeholder handling** - good models will preserve or replace appropriately
- Consider **code structure** beyond just API calls

## Questions?

See the full [README.md](README.md) for detailed documentation.
