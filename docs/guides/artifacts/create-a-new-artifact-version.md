---
description: Create a new artifact version from a single run or from a distributed process.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create new artifact versions

<head>
    <title>Create new artifacts versions from single and multiprocess Runs.</title>
</head>

Create a new artifact version with a single run, collaboratively with distributed writers, or as an incremental change to a prior version.

* **Simple**: A single run provides all the data for a new version. This is the most common case and is best suited when the run fully recreates the needed data. For example: outputting saved models or model predictions in a table for analysis.
* **Collaborative**: A set of runs collectively provides all the data for a new version. This is best suited for distributed jobs which have multiple runs generating data, often in parallel. For example: evaluating a model in a distributed manner, and outputting the predictions.
* **Incremental Change:** Add, modify, or remove a subset of files from the previous version. This is best suited when you have a large artifact composed of many underlying files. For example a dataset artifact, and you are looking to add, modify, or delete only a small number of files. This can be done within a single run or outside.

![Artifact overview diagram](/images/artifacts/incremental_artifacts_Diagram.png)

## Simple mode

Use Simple mode to log a new version of an artifact. This mode applies to the case when a single run produces all the files in the artifact.

### Create a new artifact with simple mode
Follow the procedure below to create a new artifact in simple mode:

<Tabs
  defaultValue="within"
  values={[
    {label: 'Within a run', value: 'within'},
    {label: 'Outside a run', value: 'outside'},
  ]}>
  <TabItem value="within">

1. Create with `wandb.init`. (Line 1)
2. Create an artifact object with `wandb.Artifact`. (Line 2)
3. Add files to the artifact with `.add_file`. (Line 9)
4. Log the artifacts to the run with `.log_artifact`. (Line 10)

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact(
        "artifact_name", 
        "artifact_type"
        )

    # Add Files and Assets to the artifact using 
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

  </TabItem>
  <TabItem value="outside">

Create an artifact version outside of a W&B run:

1. Create an artifact object with the W&B Public API (`wanb.Artifact`). (Line 1)
2. Add files to the artifact with `.add_file`. (Line 4)
3. Use `Artifact.save()` to create the version. (Line 5)

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# Add Files and Assets to the artifact using 
# `.add`, `.add_file`, `.add_dir`, and `.add_reference`
artifact.add_file("image1.png")
artifact.save()
```  

  </TabItem>
</Tabs>



## Collaborative Mode

Use Collaborative Mode to allow a collection of runs to collaborate on a version before committing it. Use `upsert_artifact` to add to the collaborative artifact and `finish_artifact` to finalize the commit.

:::info
1. Each run in the collection needs to be aware of the same unique ID (called `distributed_id`) in order to collaborate on the same version. By default, if present, W&B uses the run's `group` as set by `wandb.init(group=GROUP)` as the `distributed_id`.
2. There must be a final run that "commits" the version, permanently locking its state.
3. Use `upsert_artifact` to add the the collaborative artifact and `finish_artifact` to finalize the commit.
:::

### Create an artifact collaboratively across different runs

Consider the following example. Different runs (labelled below as **Run 1**, **Run 2**, and **Run 3**) add different image file (image.png) versions to the same artifact with `upsert_artifact`.


#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Add Files and Assets to the artifact using 
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image1.png")
    run.upsert_artifact(
        artifact, 
        distributed_id="my_dist_artifact"
        )     
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Add Files and Assets to the artifact using 
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image2.png")
    run.upsert_artifact(
        artifact, 
        distributed_id="my_dist_artifact"
        )
```

#### Run 3

Must run after Run 1 and Run 2 complete. The Run that calls `finish_artifact` can include files in the artifact, but does not need to.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Add Files and Assets to the artifact  
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image3.png")
    run.finish_artifact(
        artifact, 
        distributed_id="my_dist_artifact"
        )
```




## Incremental change

Use incremental artifacts to apply changes to a small subset of files without waiting for the process to re-index, download, or reference the rest of the files in an artifact. There are three types of incremental changes you can make to an artifact:

|            | Common use case |
| ----- | ----|
| add |  periodically add a new subset of files to a dataset after collecting a new batch. |
| remove  | you discovered several duplicate files and want to remove them from your artifact.| 
| modify  | you corrected annotations for a subset of files and want to replace the old files with the correct ones.|

### How to incrementally change your artifact

Follow the procedure below to incrementally change an artifact:

1. Obtain the artifact version you want to perform an incremental change on with the W&B Public API:

```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```

2. Create a draft with:

```python
draft_artifact = saved_artifact.new_draft()
```

3. Perform any incremental changes you want to see in the next version:


<Tabs
  defaultValue="add"
  values={[
    {label: 'Add', value: 'add'},
    {label: 'Remove', value: 'remove'},
    {label: 'Modify', value: 'modify'},
  ]}>
  <TabItem value="add">Add a file:

```python
draft_artifact.add_file("file_to_add.txt")
```
  </TabItem>
  <TabItem value="remove">
Remove a file:  

```python
draft_artifact.remove("file_to_remove.txt")
```
  
  </TabItem>
  <TabItem value="modify">
Modify, replace a file:

```python
draft_artifact.add_file("modified_file.txt")
```

  </TabItem>
</Tabs>

:::info
The method to add or modify an artifact are the same. Entries are replaced (as opposed to duplicated), when you pass a filename for an entry that already exists.
:::