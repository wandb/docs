---
description: Create a new artifact version from a single run or from a distributed
  process.
menu:
  default:
    identifier: create-a-new-artifact-version
    parent: artifacts
title: Create an artifact version
weight: 6
---

Create a new artifact version with a single [run](../runs/intro.md) or collaboratively with distributed runs. You can optionally create a new artifact version from a previous version known as an [incremental artifact](#create-a-new-artifact-version-from-an-existing-version).

{{% alert %}}
We recommend that you create an incremental artifact when you need to apply changes to a subset of files in an artifact, where the size of the original artifact is significantly larger.
{{% /alert %}}

## Create new artifact versions from scratch
There are two ways to create a new artifact version: from a single run and from distributed runs. They are defined as follows:


* **Single run**: A single run provides all the data for a new version. This is the most common case and is best suited when the run fully recreates the needed data. For example: outputting saved models or model predictions in a table for analysis.
* **Distributed runs**: A set of runs collectively provides all the data for a new version. This is best suited for distributed jobs which have multiple runs generating data, often in parallel. For example: evaluating a model in a distributed manner, and outputting the predictions.


W&B will create a new artifact and assign it a `v0` alias if you pass a name to the `wandb.Artifact` API that does not exist in your project. W&B checksums the contents when you log again to the same artifact. If the artifact changed, W&B saves a new version `v1`.  

W&B will retrieve an existing artifact if you pass a name and artifact type to the `wandb.Artifact` API that matches an existing artifact in your project. The retrieved artifact will have a version greater than 1. 

{{< img src="/images/artifacts/single_distributed_artifacts.png" alt="" >}}

### Single run
Log a new version of an Artifact with a single run that produces all the files in the artifact. This case occurs when a single run produces all the files in the artifact. 

Based on your use case, select one of the tabs below to create a new artifact version inside or outside of a run:

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
Create an artifact version within a W&B run:

1. Create a run with `wandb.init`. (Line 1)
2. Create a new artifact or retrieve an existing one with `wandb.Artifact` . (Line 2)
3. Add files to the artifact with `.add_file`. (Line 9)
4. Log the artifact to the run with `.log_artifact`. (Line 10)

```python showLineNumbers
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # Add Files and Assets to the artifact using
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
Create an artifact version outside of a W&B run:

1. Create a new artifact or retrieve an existing one with `wanb.Artifact`. (Line 1)
2. Add files to the artifact with `.add_file`. (Line 4)
3. Save the artifact with `.save`. (Line 5)

```python showLineNumbers
artifact = wandb.Artifact("artifact_name", "artifact_type")
# Add Files and Assets to the artifact using
# `.add`, `.add_file`, `.add_dir`, and `.add_reference`
artifact.add_file("image1.png")
artifact.save()
```    
  {{% /tab %}}
{{< /tabpane  >}}




### Distributed runs

Allow a collection of runs to collaborate on a version before committing it. This is in contrast to single run mode described above where one run provides all the data for a new version.


{{% alert %}}
1. Each run in the collection needs to be aware of the same unique ID (called `distributed_id`) in order to collaborate on the same version. By default, if present, W&B uses the run's `group` as set by `wandb.init(group=GROUP)` as the `distributed_id`.
2. There must be a final run that "commits" the version, permanently locking its state.
3. Use `upsert_artifact` to add to the collaborative artifact and `finish_artifact` to finalize the commit.
{{% /alert %}}

Consider the following example. Different runs (labelled below as **Run 1**, **Run 2**, and **Run 3**) add a different image file to the same artifact with `upsert_artifact`.


#### Run 1:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Add Files and Assets to the artifact using
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image1.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 2:

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Add Files and Assets to the artifact using
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image2.png")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

#### Run 3

Must run after Run 1 and Run 2 complete. The Run that calls `finish_artifact` can include files in the artifact, but does not need to.

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # Add Files and Assets to the artifact
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image3.png")
    run.finish_artifact(artifact, distributed_id="my_dist_artifact")
```




## Create a new artifact version from an existing version

Add, modify, or remove a subset of files from a previous artifact version without the need to re-index the files that didn't change. Adding, modifying, or removing a subset of files from a previous artifact version creates a new artifact version known as an *incremental artifact*.

{{< img src="/images/artifacts/incremental_artifacts.png" alt="" >}}

Here are some scenarios for each type of incremental change you might encounter:

- add: you periodically add a new subset of files to a dataset after collecting a new batch.
- remove: you discovered several duplicate files and want to remove them from your artifact.
- update: you corrected annotations for a subset of files and want to replace the old files with the correct ones.

You could create an artifact from scratch to perform the same function as an incremental artifact. However, when you create an artifact from scratch, you will need to have all the contents of your artifact on your local disk. When making an incremental change, you can add, remove, or modify a single file without changing the files from a previous artifact version.


{{% alert %}}
You can create an incremental artifact within a single run or with a set of runs (distributed mode).
{{% /alert %}}


Follow the procedure below to incrementally change an artifact:

1. Obtain the artifact version you want to perform an incremental change on:


{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}

```python
saved_artifact = run.use_artifact("my_artifact:latest")
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")
```  
  {{% /tab %}}
{{< /tabpane >}}



2. Create a draft with:

```python
draft_artifact = saved_artifact.new_draft()
```

3. Perform any incremental changes you want to see in the next version. You can either add, remove, or modify an existing entry.

Select one of the tabs for an example on how to perform each of these changes:


{{< tabpane text=true >}}
  {{% tab header="Add" %}}
Add a file to an existing artifact version with the `add_file` method:

```python
draft_artifact.add_file("file_to_add.txt")
```

{{% alert %}}
You can also add multiple files by adding a directory with the `add_dir` method.
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="Remove" %}}
Remove a file from an existing artifact version with the `remove` method:

```python
draft_artifact.remove("file_to_remove.txt")
```

{{% alert %}}
You can also remove multiple files with the `remove` method by passing in a directory path.
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="Modify" %}}
Modify or replace contents by removing the old contents from the draft and adding the new contents back in:

```python
draft_artifact.remove("modified_file.txt")
draft_artifact.add_file("modified_file.txt")
```  
  {{% /tab %}}
{{< /tabpane >}}


<!-- {{% alert %}}
The method to add or modify an artifact are the same. Entries are replaced (as opposed to duplicated), when you pass a filename for an entry that already exists.
{{% /alert %}} -->

4. Lastly, log or save your changes. The following tabs show you how to save your changes inside and outside of a W&B run. Select the tab that is appropriate for your use case:

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
```python
run.log_artifact(draft_artifact)
```

  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
```python
draft_artifact.save()
```  
  {{% /tab %}}
{{< /tabpane >}}


Putting it all together, the code examples above look like: 

{{< tabpane text=true >}}
  {{% tab header="Inside a run" %}}
```python
with wandb.init(job_type="modify dataset") as run:
    saved_artifact = run.use_artifact(
        "my_artifact:latest"
    )  # fetch artifact and input it into your run
    draft_artifact = saved_artifact.new_draft()  # create a draft version

    # modify a subset of files in the draft version
    draft_artifact.add_file("file_to_add.txt")
    draft_artifact.remove("dir_to_remove/")
    run.log_artifact(
        artifact
    )  # log your changes to create a new version and mark it as output to your run
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
```python
client = wandb.Api()
saved_artifact = client.artifact("my_artifact:latest")  # load your artifact
draft_artifact = saved_artifact.new_draft()  # create a draft version

# modify a subset of files in the draft version
draft_artifact.remove("deleted_file.txt")
draft_artifact.add_file("modified_file.txt")
draft_artifact.save()  # commit changes to the draft
```  
  {{% /tab %}}
{{< /tabpane >}}
