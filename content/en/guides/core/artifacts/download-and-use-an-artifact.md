---
description: Download and use Artifacts from multiple projects.
menu:
  default:
    identifier: download-and-use-an-artifact
    parent: artifacts
title: Download and use artifacts
weight: 3
---

Download and use an artifact that is already stored on the W&B server or construct an artifact object and pass it in to for de-duplication as necessary.

{{% alert %}}
Team members with view-only seats cannot download artifacts.
{{% /alert %}}


### Download and use an artifact stored on W&B

Download and use an artifact stored in W&B either inside or outside of a W&B Run. Use the Public API ([`wandb.Api`]({{< relref "/ref/python/public-api/api.md" >}})) to export (or update data) already saved in W&B. For more information, see the W&B [Public API Reference guide]({{< relref "/ref/python/public-api/index.md" >}}).

{{< tabpane text=true >}}
  {{% tab header="During a run" %}}
First, import the W&B Python SDK. Next, create a W&B [Run]({{< relref "/ref/python/sdk/classes/run.md" >}}):

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
```

Indicate the artifact you want to use with the [`use_artifact`]({{< relref "/ref/python/sdk/classes/run.md#use_artifact" >}}) method. This returns a run object. In the proceeding code snippet specifies an artifact called `'bike-dataset'` with the alias `'latest'`:

```python
artifact = run.use_artifact("bike-dataset:latest")
```

Use the object returned to download all the contents of the artifact:

```python
datadir = artifact.download()
```

You can optionally pass a path to the root parameter to download the contents of the artifact to a specific directory. For more information, see the [Python SDK Reference Guide]({{< relref "/ref/python/sdk/classes/artifact.md#download" >}}).

Use the [`get_path`]({{< relref "/ref/python/sdk/classes/artifact.md#get_path" >}}) method to download only subset of files:

```python
path = artifact.get_path(name)
```

This fetches only the file at the path `name`. It returns an `Entry` object with the following methods:

* `Entry.download`: Downloads file from the artifact at path `name`
* `Entry.ref`: If `add_reference` stored the entry as a reference, returns the URI

References that have schemes that W&B knows how to handle get downloaded just like artifact files. For more information, see [Track external files]({{< relref "/guides/core/artifacts/track-external-files.md" >}}).  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
First, import the W&B SDK. Next, create an artifact from the Public API Class. Provide the entity, project, artifact, and alias associated with that artifact:

```python
import wandb

api = wandb.Api()
artifact = api.artifact("entity/project/artifact:alias")
```

Use the object returned to download the contents of the artifact:

```python
artifact.download()
```

You can optionally pass a path the `root` parameter to download the contents of the artifact to a specific directory. For more information, see the [API Reference Guide]({{< relref "/ref/python/sdk/classes/artifact.md#download" >}}).  
  {{% /tab %}}
  {{% tab header="W&B CLI" %}}
Use the `wandb artifact get` command to download an artifact from the W&B server.

```
$ wandb artifact get project/artifact:alias --root mnist/
```  
  {{% /tab %}}
{{< /tabpane >}}


### Partially download an artifact

You can optionally download part of an artifact based on a prefix. Using the `path_prefix` parameter, you can download a single file or the content of a sub-folder.

```python
artifact = run.use_artifact("bike-dataset:latest")

artifact.download(path_prefix="bike.png") # downloads only bike.png
```

Alternatively, you can download files from a certain directory:

```python
artifact.download(path_prefix="images/bikes/") # downloads files in the images/bikes directory
```
### Use an artifact from a different project

Specify the name of artifact along with its project name to reference an artifact. You can also reference artifacts across entities by specifying the name of the artifact with its entity name.

The following code example demonstrates how to query an artifact from another project as input to the current W&B run.

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
# Query W&B for an artifact from another project and mark it
# as an input to this run.
artifact = run.use_artifact("my-project/artifact:alias")

# Use an artifact from another entity and mark it as an input
# to this run.
artifact = run.use_artifact("my-entity/my-project/artifact:alias")
```

### Construct and use an artifact simultaneously

Simultaneously construct and use an artifact. Create an artifact object and pass it to use_artifact. This creates an artifact in W&B if it does not exist yet. The [`use_artifact`]({{< relref "/ref/python/sdk/classes/run.md#use_artifact" >}}) API is idempotent, so you can call it as many times as you like.

```python
import wandb

artifact = wandb.Artifact("reference model")
artifact.add_file("model.h5")
run.use_artifact(artifact)
```

For more information about constructing an artifact, see [Construct an artifact]({{< relref "/guides/core/artifacts/construct-an-artifact.md" >}}).