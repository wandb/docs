---
title: Query Expression Language
menu:
    reference:
        identifier: query-panel
        parent: reference
        weight: 4
---

The W&B Query Expression Language lets you programmatically analyze and visualize your ML experiments directly in the W&B UI. Transform raw experiment data into actionable insights using powerful query operations.

## Important: Where Query Expressions Run

**Query Expressions are NOT local code!** They are typed directly into the W&B web interface, not in your Python/JavaScript files.

## Getting Started

### Step 1: Log Data to W&B (Local Code)

First, you need data in W&B to query. This requires the W&B Python SDK:

```bash
pip install wandb  # Only installation needed - for logging data
```

```python
# your_training_script.py - runs locally
import wandb

wandb.init(project="my-ml-project")
wandb.log({"loss": 0.5, "accuracy": 0.85})
wandb.finish()
```

### Step 2: Query Your Data (W&B Web UI)

After logging runs, analyze them in the W&B web interface:

1. **Open your browser** and go to [wandb.ai](https://wandb.ai)
2. **Navigate to your project** (e.g., `wandb.ai/your-username/my-ml-project`)
3. **Click "+ Add Panel"** → Select **"Query Panel"**
4. **Type expressions in the web editor** (NOT in your local code):
   ```typescript
   // This is typed into the wandb.ai interface
   runs.map(r => runSummary(r).accuracy).avg()
   ```
5. **See results instantly** as charts or tables in your browser

## Complete Example: Finding Your Best Model

Here's what you would type **in the W&B Query Panel editor** to analyze a hyperparameter sweep:

```typescript
// Remember: This is typed into the Query Panel at wandb.ai
// NOT in your local code files!

// Step 1: Filter to successful runs from your latest sweep
const validRuns = runs
  .filter(r => r.state === "finished")
  .filter(r => runConfig(r).sweep_id === "sweep_2024_01")

// Step 2: Extract key metrics and configurations
const runAnalysis = validRuns.map(r => ({
  name: r.name,
  accuracy: runSummary(r).best_accuracy,
  loss: runSummary(r).final_loss,
  learning_rate: runConfig(r).learning_rate,
  batch_size: runConfig(r).batch_size,
  training_time: r.duration
}))

// Step 3: Find the best run
const bestRun = validRuns
  .reduce((best, current) => 
    runSummary(current).best_accuracy > runSummary(best).best_accuracy 
      ? current 
      : best
  )

// Step 4: Calculate statistics across all runs
const stats = {
  avg_accuracy: validRuns.map(r => runSummary(r).best_accuracy).avg(),
  std_accuracy: validRuns.map(r => runSummary(r).best_accuracy).std(),
  total_compute_hours: validRuns.map(r => r.duration).sum() / 3600
}

// Step 5: Group by hyperparameter to find optimal values
const byLearningRate = validRuns
  .groupby(r => runConfig(r).learning_rate)
  .map(group => ({
    lr: group.key,
    avg_accuracy: group.values.map(r => runSummary(r).best_accuracy).avg(),
    num_runs: group.values.length
  }))
```

## Core Concepts

### Chainable Operations
All operations can be chained together for powerful data transformations:

```typescript
runs
  .filter(/* select runs */)
  .map(/* transform data */)
  .groupby(/* organize results */)
  .sort(/* order output */)
```

### Type Safety
The expression language is fully typed, providing autocomplete and validation as you write queries.


## [Operations](operations/)
Functions for querying and manipulating W&B data:
- [Run Operations](operations/run-operations.md) - Query and manipulate runs
- [Artifact Operations](operations/artifact-operations.md) - Work with artifacts

## [Data Types](data-types/)
Core type definitions:
- [Run](data-types/run.md) - Experiment runs with metadata and metrics
- [Artifact](data-types/artifact.md) - Versioned files and directories
- [ArtifactType](data-types/artifacttype.md) - Artifact type definitions
- [ArtifactVersion](data-types/artifactversion.md) - Specific artifact versions
- [ConfigDict](data-types/configdict.md) - Configuration parameters
- [SummaryDict](data-types/summarydict.md) - Summary metrics from runs
- [Table](data-types/table.md) - Tabular data structure
- [User](data-types/user.md) - User account information
- [Project](data-types/project.md) - Project metadata
- [Entity](data-types/entity.md) - Team or user organization

## Common Patterns

The following examples show Query Expressions you would type in the W&B web UI:

### Compare Model Architectures
```typescript
// Type this in the Query Panel at wandb.ai
// Group runs by model type and compare average performance
runs
  .groupby(r => runConfig(r).model_type)
  .map(g => ({
    model: g.key,
    avg_accuracy: g.values.map(r => runSummary(r).accuracy).avg(),
    best_accuracy: g.values.map(r => runSummary(r).accuracy).max(),
    training_hours: g.values.map(r => r.duration).sum() / 3600
  }))
  .sort((a, b) => b.avg_accuracy - a.avg_accuracy)
```

### Track Experiment Progress
```typescript
// Monitor ongoing experiments
runs
  .filter(r => r.state === "running")
  .map(r => ({
    name: r.name,
    progress: runSummary(r).epoch / runConfig(r).total_epochs,
    current_loss: runSummary(r).loss,
    eta_minutes: (r.duration / runSummary(r).epoch) * 
                 (runConfig(r).total_epochs - runSummary(r).epoch) / 60
  }))
```

### Find Optimal Hyperparameters
```typescript
// Identify best performing hyperparameter combinations
runs
  .filter(r => runSummary(r).val_accuracy > 0.85)
  .map(r => ({
    accuracy: runSummary(r).val_accuracy,
    lr: runConfig(r).learning_rate,
    batch_size: runConfig(r).batch_size,
    optimizer: runConfig(r).optimizer
  }))
  .sort((a, b) => b.accuracy - a.accuracy)
  .slice(0, 10)  // Top 10 configurations
```

## See Also

- [Query Panels Guide](/guides/models/app/features/panels/query-panels/) - Visual tutorial with screenshots
- [W&B Python SDK](/ref/python/) - For logging runs and artifacts