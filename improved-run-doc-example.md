---
title: Run Operations
description: Operations for querying and manipulating W&B runs
menu:
  reference:
    parent: qel
    weight: 20
---

# Run Operations

The `run` type represents a Weights & Biases experiment run with associated configuration, metrics, and artifacts. These operations allow you to query and manipulate run data within query panels.

## Data Type

### Run

A W&B run contains:
- **Metadata**: ID, name, project, entity, creation time
- **Configuration**: Hyperparameters and settings
- **Metrics**: Summary values and time-series history
- **Artifacts**: Input and output artifacts with versions
- **State**: Current run status (running, finished, failed, crashed)

## Chainable Operations

### run-config

Returns the configuration dictionary containing hyperparameters and settings used to initialize the run.

#### Syntax
```
run-config(run: Run) → ConfigDict
```

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `run` | `Run` | Yes | The W&B run to extract configuration from |

#### Returns
`ConfigDict` - Dictionary containing all hyperparameters and configuration settings

#### Examples

##### Basic Usage
```javascript
// Get configuration from a specific run
config = run-config(run)
```

##### Filter by Configuration
```javascript
// Find runs with specific learning rate
runs.filter(r => run-config(r).learning_rate > 0.01)
```

##### Compare Configurations
```javascript
// Compare configs between two runs
config1 = run-config(run1)
config2 = run-config(run2)
diff = compareConfigs(config1, config2)
```

---

### run-summary

Returns the summary metrics representing final or best values logged during a run.

#### Syntax
```
run-summary(run: Run) → SummaryDict
```

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `run` | `Run` | Yes | The W&B run to extract summary from |

#### Returns
`SummaryDict` - Dictionary of summary metrics (e.g., final accuracy, best validation loss)

#### Examples

##### Get Final Metrics
```javascript
summary = run-summary(run)
accuracy = summary.accuracy
val_loss = summary.best_val_loss
```

##### Find Best Run
```javascript
// Find run with highest accuracy
bestRun = runs.reduce((best, run) => 
  run-summary(run).accuracy > run-summary(best).accuracy ? run : best
)
```

---

### run-createdAt

Returns the timestamp when the run was created.

#### Syntax
```
run-createdAt(run: Run) → DateTime
```

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `run` | `Run` | Yes | The W&B run to get creation time from |

#### Returns
`DateTime` - The creation timestamp

#### Examples

##### Filter Recent Runs
```javascript
// Get runs from last 7 days
recentRuns = runs.filter(r => 
  run-createdAt(r) > Date.now() - 7*24*60*60*1000
)
```

##### Sort Chronologically
```javascript
// Sort runs by creation time
sortedRuns = runs.sort((a, b) => 
  run-createdAt(a) - run-createdAt(b)
)
```

---

### run-loggedArtifactVersion

Retrieves a specific artifact version that was logged (output) by the run.

#### Syntax
```
run-loggedArtifactVersion(run: Run, artifactVersionName: String) → ArtifactVersion?
```

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `run` | `Run` | Yes | The W&B run that logged the artifact |
| `artifactVersionName` | `String` | Yes | Artifact identifier as "name:alias" (e.g., "model:latest") |

#### Returns
`ArtifactVersion` or `null` - The artifact version object if found

#### Examples

##### Get Model Artifact
```javascript
// Retrieve the latest model from a run
model = run-loggedArtifactVersion(run, "model:latest")
if (model) {
  console.log(`Model version: ${model.version}`)
  console.log(`Model size: ${model.size}`)
}
```

##### Validate Dataset Version
```javascript
// Ensure run used correct dataset
dataset = run-loggedArtifactVersion(run, "training-data:v3")
if (!dataset) {
  alert("Warning: Run did not use expected dataset version")
}
```

---

### run-usedArtifactVersions

Returns all artifact versions that were used as inputs by the run.

#### Syntax
```
run-usedArtifactVersions(run: Run) → List<ArtifactVersion>
```

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `run` | `Run` | Yes | The W&B run to get input artifacts from |

#### Returns
`List<ArtifactVersion>` - List of all input artifact versions

#### Examples

##### List All Inputs
```javascript
// Get all artifacts used by the run
inputs = run-usedArtifactVersions(run)
inputs.forEach(artifact => {
  console.log(`Used: ${artifact.name}:${artifact.version}`)
})
```

##### Check Dependencies
```javascript
// Verify run used specific preprocessing pipeline
inputs = run-usedArtifactVersions(run)
hasPreprocessing = inputs.some(a => 
  a.name === "preprocessing-pipeline"
)
```

## List Operations

List operations work on collections of runs and support aggregation.

### Aggregation Functions

When working with lists of runs, you can use aggregation functions:

- `count()` - Count number of runs
- `avg()` - Average of numeric values
- `max()` - Maximum value
- `min()` - Minimum value
- `sum()` - Sum of values
- `stddev()` - Standard deviation

#### Example: Aggregate Metrics
```javascript
// Average accuracy across all runs
runs.map(r => run-summary(r).accuracy).avg()

// Maximum learning rate used
runs.map(r => run-config(r).learning_rate).max()

// Count runs by status
runs.groupBy(r => run-summary(r).state).count()
```

## Common Patterns

### Filtering Runs
```javascript
// Successful runs with high accuracy
successfulRuns = runs.filter(r => 
  run-summary(r).state === "finished" &&
  run-summary(r).accuracy > 0.9
)
```

### Comparing Runs
```javascript
// Find runs with different configs but similar performance
baseline = runs[0]
similar = runs.filter(r => 
  Math.abs(run-summary(r).accuracy - run-summary(baseline).accuracy) < 0.01 &&
  run-config(r).model_type !== run-config(baseline).model_type
)
```

### Time-based Analysis
```javascript
// Group runs by day
runsByDay = runs.groupBy(r => 
  run-createdAt(r).toDateString()
)
```

## Related Operations

- [artifact operations](./artifact.md) - Work with artifact data
- [project operations](./project.md) - Query project-level data
- [user operations](./user.md) - Get user information
- [table operations](./table.md) - Manipulate tabular data

## See Also

- [Query Panels Guide](/guides/models/app/features/panels/query-panels/)
- [W&B Run API Reference](/ref/python/run)
- [Artifacts Documentation](/guides/artifacts)
