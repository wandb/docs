---
title: Run Operations
---

Operations for querying and manipulating W&B runs

## runConfig

```typescript
runConfig(run): ConfigDict
```

Extracts the configuration dictionary ([`ConfigDict`](../data-types/configdict.md)) from a W&B run.

The configuration contains hyperparameters and settings used when the run was initialized.
This is particularly useful for comparing configurations across experiments or filtering
runs based on specific parameter values.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run object to extract configuration from |

#### Example: Basic Configuration Access
```typescript
const config = runConfig(myRun);
console.log(config.learning_rate); // 0.001
console.log(config.batch_size);    // 32
```

#### Example: Filtering Runs by Configuration
```typescript
// Find all runs with learning rate > 0.01
const highLRRuns = runs.filter(run => {
  const config = runConfig(run);
  return config.learning_rate > 0.01;
});
```

#### Example: Comparing Configurations
```typescript
const config1 = runConfig(baseline);
const config2 = runConfig(experiment);
const differences = Object.keys(config1).filter(key => 
  config1[key] !== config2[key]
);
```

#### See Also

 - [runSummary](#runsummary) - For accessing summary metrics
 - runHistory - For accessing time-series data

___

## runCreatedAt

```typescript
runCreatedAt(run): Date
```

Gets the creation timestamp of a W&B run.

Returns when the run was first initialized. Useful for chronological sorting,
filtering by date ranges, or analyzing experiment progression over time.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run to get creation time from |

#### Example: Filter Recent Runs
```typescript
const oneWeekAgo = new Date();
oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);

const recentRuns = runs.filter(run => 
  runCreatedAt(run) > oneWeekAgo
);
```

#### Example: Sort Chronologically
```typescript
const sortedRuns = runs.sort((a, b) => 
  runCreatedAt(a).getTime() - runCreatedAt(b).getTime()
);
```

#### Example: Group by Date
```typescript
const runsByDate = runs.reduce((groups, run) => {
  const date = runCreatedAt(run).toDateString();
  groups[date] = groups[date] || [];
  groups[date].push(run);
  return groups;
}, {});
```

#### See Also

 - [runHeartbeatAt](#runheartbeatat) - For last activity time
 - [runRuntime](#runruntime) - For run duration

___

## runHeartbeatAt

```typescript
runHeartbeatAt(run): Date
```

Gets the last heartbeat timestamp of a W&B run.

The heartbeat indicates when the run last sent data to W&B. For active runs,
this is continuously updated. For finished runs, it shows the completion time.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run to get heartbeat from |

#### Example: Check if Run is Active
```typescript
const isActive = (run: Run) => {
  const lastHeartbeat = runHeartbeatAt(run);
  const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
  return lastHeartbeat > fiveMinutesAgo;
};
```

#### See Also

 - [runCreatedAt](#runcreatedat) - For run start time
 - [runRuntime](#runruntime) - For total run duration

___

## runJobType

```typescript
runJobType(run): string | undefined
```

Gets the job type of a run.

Returns the job type classification (e.g., "train", "eval", "sweep") if set.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run to get job type from |

#### Example: Filter by Job Type
```typescript
const trainingRuns = runs.filter(run => 
  runJobType(run) === "train"
);
const evalRuns = runs.filter(run => 
  runJobType(run) === "eval"
);
```

#### See Also

[Run](../data-types/run.md) - Run type definition

___

## runLoggedArtifactVersion

```typescript
runLoggedArtifactVersion(run, artifactVersionName): ArtifactVersion | undefined
```

Gets a specific artifact version logged (output) by a run.

Artifacts in W&B are versioned files or directories that track model checkpoints,
datasets, or other outputs. This function retrieves a specific artifact version
that was created/logged during the run's execution.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run that logged the artifact |
| `artifactVersionName` | `string` | Artifact identifier as "name:alias" (e.g., "model:latest", "dataset:v2") |

#### Example: Get Latest Model
```typescript
const model = runLoggedArtifactVersion(run, "model:latest");
if (model) {
  console.log(Model version: v${model.version});
  console.log(Model size: ${model.size} bytes);
  console.log(Created: ${model.createdAt});
}
```

#### Example: Verify Output Artifacts
```typescript
const requiredOutputs = ["model:latest", "evaluation:latest"];
const missing = requiredOutputs.filter(name => 
  !runLoggedArtifactVersion(run, name)
);
if (missing.length > 0) {
  console.warn(Missing outputs: ${missing.join(", ")});
}
```

#### See Also

 - [runLoggedArtifactVersions](#runloggedartifactversions) - Get all logged artifacts
 - [runUsedArtifactVersions](#runusedartifactversions) - Get input artifacts

___

## runLoggedArtifactVersions

```typescript
runLoggedArtifactVersions(run): ArtifactVersion[]
```

Gets all artifact versions logged (output) by a run.

Returns a complete list of all artifacts created during the run's execution,
including models, datasets, checkpoints, and other outputs.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run to get logged artifacts from |

#### Example: List All Outputs
```typescript
const outputs = runLoggedArtifactVersions(run);
outputs.forEach(artifact => {
  console.log(Logged: ${artifact.name}:${artifact.alias});
  console.log(  Version: ${artifact.version});
  console.log(  Size: ${artifact.size} bytes);
});
```

#### Example: Count Output Types
```typescript
const outputs = runLoggedArtifactVersions(run);
const modelCount = outputs.filter(a => a.name.includes("model")).length;
const dataCount = outputs.filter(a => a.name.includes("data")).length;
console.log(Models: ${modelCount}, Datasets: ${dataCount});
```

#### See Also

 - [runLoggedArtifactVersion](#runloggedartifactversion) - Get specific artifact
 - [runUsedArtifactVersions](#runusedartifactversions) - Get input artifacts

___

## runName

```typescript
runName(run): string
```

Gets the name/ID of a run.

Returns the unique run name (ID) assigned by W&B or set by the user.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run to get name from |

#### Example: Display Run Names
```typescript
runs.forEach(run => {
  console.log(Run: ${runName(run)});
});
```

#### See Also

[Run](../data-types/run.md) - Run type definition

___

## runRuntime

```typescript
runRuntime(run): number
```

Calculates the runtime duration of a W&B run in seconds.

Returns the total execution time from creation to last heartbeat.
For active runs, this represents the current runtime.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run to calculate runtime for |

#### Example: Display Runtime
```typescript
const runtime = runRuntime(myRun);
const hours = Math.floor(runtime / 3600);
const minutes = Math.floor((runtime % 3600) / 60);
console.log(Runtime: ${hours}h ${minutes}m);
```

#### Example: Filter Long-Running Experiments
```typescript
const longRuns = runs.filter(run => 
  runRuntime(run) > 3600 // More than 1 hour
);
```

#### See Also

 - [runCreatedAt](#runcreatedat) - For start time
 - [runHeartbeatAt](#runheartbeatat) - For end/current time

___

## runSummary

```typescript
runSummary(run): SummaryDict
```

Retrieves summary metrics ([`SummaryDict`](../data-types/summarydict.md)) from a W&B run.

Summary metrics represent the final or best values logged during a run's execution,
such as final accuracy, best validation loss, or total training time. These are
scalar values that summarize the run's overall performance.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run to extract summary from |

#### Example: Accessing Final Metrics
```typescript
const summary = runSummary(myRun);
console.log(Final accuracy: ${summary.accuracy});
console.log(Best validation loss: ${summary.best_val_loss});
console.log(Training time: ${summary.training_time_seconds});
```

#### Example: Finding Best Performing Run
```typescript
const bestRun = runs.reduce((best, current) => {
  const bestSummary = runSummary(best);
  const currentSummary = runSummary(current);
  return currentSummary.accuracy > bestSummary.accuracy ? current : best;
});
```

#### Example: Filtering by Performance Threshold
```typescript
const goodRuns = runs.filter(run => {
  const summary = runSummary(run);
  return summary.accuracy > 0.95 && summary.val_loss < 0.1;
});
```

#### See Also

 - [runConfig](#runconfig) - For configuration parameters
 - runHistory - For time-series metrics

___

## runUsedArtifactVersions

```typescript
runUsedArtifactVersions(run): ArtifactVersion[]
```

Gets all artifact versions used (input) by a run.

Returns artifacts that were consumed as inputs during the run's execution,
such as training datasets, pretrained models, or configuration files.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run to get used artifacts from |

#### Example: List Input Dependencies
```typescript
const inputs = runUsedArtifactVersions(run);
console.log("Run dependencies:");
inputs.forEach(artifact => {
  console.log(- ${artifact.name}:${artifact.version});
});
```

#### Example: Verify Dataset Version
```typescript
const inputs = runUsedArtifactVersions(run);
const dataset = inputs.find(a => a.name === "training-data");
if (dataset && dataset.version !== 3) {
  console.warn(Using outdated dataset v${dataset.version});
}
```

#### See Also

 - [runLoggedArtifactVersions](#runloggedartifactversions) - Get output artifacts
 - [runLoggedArtifactVersion](#runloggedartifactversion) - Get specific output

___

## runUser

```typescript
runUser(run): User
```

Gets the user who created the run.

Returns the W&B user object associated with the run, useful for filtering
by user or analyzing team member contributions.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `run` | [`Run`](../data-types/run.md) | The W&B run to get user from |

#### Example: Filter by User
```typescript
const myRuns = runs.filter(run => 
  runUser(run).username === "john_doe"
);
```

#### Example: Group Runs by User
```typescript
const runsByUser = runs.reduce((groups, run) => {
  const user = runUser(run).username;
  groups[user] = groups[user] || [];
  groups[user].push(run);
  return groups;
}, {});
```

#### See Also

[User](../data-types/user.md) - User type definition
