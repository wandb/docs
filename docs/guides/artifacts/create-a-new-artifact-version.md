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

Create a new artifact version with a single [run](../runs/intro.md) or collaboratively with distributed runs. You can optionally create a new artifact version from a previous version known as an [incremental artifact](#create-a-new-version-from-an-existing-version).




<!-- ![Artifact overview diagram](/images/artifacts/incremental_artifacts_Diagram.png) -->
## Create new artifact versions from scratch
There are two ways to create a new artifact version: from a single run and from distributed runs. They are defined as follows:


* **Single run**: A single run provides all the data for a new version. This is the most common case and is best suited when the run fully recreates the needed data. For example: outputting saved models or model predictions in a table for analysis.
* **Distributed runs**: A set of runs collectively provides all the data for a new version. This is best suited for distributed jobs which have multiple runs generating data, often in parallel. For example: evaluating a model in a distributed manner, and outputting the predictions.

### Single run
Use a single run to log a new version of an artifact. This case occurs when a single run produces all the files in the artifact. 

Select one of the tabs below to view the steps to create a new artifact version inside or outside of a run:

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



### Distributed runs

Use a set of runs to create an artifact version. This is in contrast to single run mode described above where one run provides all the data for a new version.

:::info
1. Each run in the collection needs to be aware of the same unique ID (called `distributed_id`) in order to collaborate on the same version. By default, if present, W&B uses the run's `group` as set by `wandb.init(group=GROUP)` as the `distributed_id`.
2. There must be a final run that "commits" the version, permanently locking its state.
3. Use `upsert_artifact` to add the the collaborative artifact and `finish_artifact` to finalize the commit.
:::

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




## Create a new artifact version from an existing version

Add, modify, or remove a subset of files from a previous artifact version without waiting for a process to re-index, download, or reference the rest of the files in an artifact. Adding, modifying, or removing a subset of files from a previous artifact version creates a new artifact version known as an *incremental artifact*.

You could create an artifact from scratch to perform the same function as an incremental artifact. However, when you create an artifact from scratch, you will need to have all the contents of your artifact on your local disk. When making an incremental change, you can add, remove, or modify a single file without changing the files from a previous artifact version.


:::tip
Incremental artifacts are particularly useful for workflows that require you to apply changes to a subset of files in an artifact that are large (for example datasets). Use incremental artifacts to avoid loading all of an artifact's contents onto your local disk.
:::

:::info
You can create an incremental artifact within a single run or with a set of runs (distributed mode).
:::


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

3. Perform any incremental changes you want to see in the next version. You can either add, remove, or modify an existing artifact version.

Select one of the tabs for an example on how to perform each of these changes:


<Tabs
  defaultValue="add"
  values={[
    {label: 'Add', value: 'add'},
    {label: 'Remove', value: 'remove'},
    {label: 'Modify', value: 'modify'},
  ]}>
  <TabItem value="add">

Add a file to an existing artifact version with the `add_file` method:

```python
draft_artifact.add_file("file_to_add.txt")
```

:::note
You can also add multiple files by adding a directory with the `add_dir` method.
:::

  </TabItem>
  <TabItem value="remove">

Remove a file to an existing artifact version with the `remove` method:

```python
draft_artifact.remove("file_to_remove.txt")
```

:::note
You can also remove multiple files with the `remove` method by passing in a directory path.
:::

  </TabItem>
  <TabItem value="modify">
  
Modify or replace a file with the `add_file` method:

```python
draft_artifact.add_file("modified_file.txt")
```

  </TabItem>
</Tabs>

:::tip
The method to add or modify an artifact are the same. Entries are replaced (as opposed to duplicated), when you pass a filename for an entry that already exists.
:::

4. Lastly, log or save your changes. The following tabs show you how to save your changes inside and outside of a W&B run. Select the tab that is appropriate for your use case:


<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside of a run', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
run.log_artifact(draft_artifact)
```

  </TabItem>
  <TabItem value="outside">


```python
draft_artifact.save()
```

  </TabItem>
</Tabs>


Putting it all together, the code examples above look like: 

<Tabs
  defaultValue="inside"
  values={[
    {label: 'Inside a run', value: 'inside'},
    {label: 'Outside of a run', value: 'outside'},
  ]}>
  <TabItem value="inside">

```python
with wandb.init() as run:
	saved_artifact = run.use_artifact("my_artifact:latest") # fetch artifact and input it into your run
	draft_artifact = saved_artifact.new_draft() # create a draft version

	# modify a subset of files in the draft version
	draft_artifact.add_file("file_to_add.txt")
	draft_artifact.remove('dir_to_remove/') 
	draft_artifact.add_file("file_to_replace.txt")  
	run.log_artifact(artifact) # log your changes to create a new version and mark it as output to your run
```

  </TabItem>
  <TabItem value="outside">


```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest") # load your artifact
draft_artifact = saved_artifact.new_draft() # create a draft version

# modify a subset of files in the draft version
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save() # commit changes to the draft
```

  </TabItem>
</Tabs>