---
description: Create, construct a W&B Artifact. Learn how to add one or more files
  or a URI reference to an Artifact.
menu:
  default:
    identifier: construct-an-artifact
    parent: artifacts
title: Create an artifact
weight: 2
---

Use the W&B Python SDK to construct artifacts from [W&B Runs]({{< relref "/ref/python/sdk/classes/run.md" >}}). You can add [files, directories, URIs, and files from parallel runs to artifacts]({{< relref "#add-files-to-an-artifact" >}}). After you add a file to an artifact, save the artifact to the W&B Server or [your own private server]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}).

For information on how to track external files, such as files stored in Amazon S3, see the [Track external files]({{< relref "./track-external-files.md" >}}) page.

## How to construct an artifact

Construct a [W&B Artifact]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) in three steps:

### 1. Create an artifact Python object with `wandb.Artifact()`

Initialize the [`wandb.Artifact()`]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) class to create an artifact object. Specify the following parameters:

* **Name**: Specify a name for your artifact. The name should be unique, descriptive, and easy to remember. Use an artifacts name to both: identify the artifact in the W&B App UI and when you want to use that artifact.
* **Type**: Provide a type. The type should be simple, descriptive and correspond to a single step of your machine learning pipeline. Common artifact types include `'dataset'` or `'model'`.


{{% alert %}}
The "name" and "type" you provide is used to create a directed acyclic graph. This means you can view the lineage of an artifact on the W&B App. 

See the [Explore and traverse artifact graphs]({{< relref "./explore-and-traverse-an-artifact-graph.md" >}}) for more information.
{{% /alert %}}


{{% alert color="secondary" %}}
Artifacts can not have the same name, even if you specify a different type for the types parameter. In other words, you can not create an artifact named `cats` of type `dataset` and another artifact with the same name of type `model`.
{{% /alert %}}

You can optionally provide a description and metadata when you initialize an artifact object. For more information on available attributes and parameters, see the [`wandb.Artifact`]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) Class definition in the Python SDK Reference Guide.

The proceeding example demonstrates how to create a dataset artifact:

```python
import wandb

artifact = wandb.Artifact(name="<replace>", type="<replace>")
```

Replace the string arguments in the preceding code snippet with your own name and type.

### 2. Add one more files to the artifact

Add files, directories, external URI references (such as Amazon S3) and more with artifact methods. For example, to add a single text file, use the [`add_file`]({{< relref "/ref/python/sdk/classes/artifact.md#add_file" >}}) method:

```python
artifact.add_file(local_path="hello_world.txt", name="optional-name")
```

You can also add multiple files with the [`add_dir`]({{< relref "/ref/python/sdk/classes/artifact.md#add_dir" >}}) method. To add files, see [Update an artifact]({{< relref "./update-an-artifact.md" >}}).

### 3. Save your artifact to the W&B server

Finally, save your artifact to the W&B server. Artifacts are associated with a run. Therefore, use a run objects [`log_artifact()`]({{< relref "/ref/python/sdk/classes/run.md#log_artifact" >}}) method to save the artifact.

```python
# Create a W&B Run. Replace 'job-type'.
run = wandb.init(project="artifacts-example", job_type="job-type")

run.log_artifact(artifact)
```

You can optionally construct an artifact outside of a W&B run. For more information, see [Track external files]({{< relref "./track-external-files.md" >}}).

{{% alert color="secondary" %}}
Calls to `log_artifact` are performed asynchronously for performant uploads. This can cause surprising behavior when logging artifacts in a loop. For example:

```python
for i in range(10):
    a = wandb.Artifact(
        "race",
        type="dataset",
        metadata={
            "index": i,
        },
    )
    # ... add files to artifact a ...
    run.log_artifact(a)
```

The artifact version **v0** is NOT guaranteed to have an index of 0 in its metadata, as the artifacts may be logged in an arbitrary order.
{{% /alert %}}

## Add files to an artifact

The following sections demonstrate how to construct artifacts with different file types and from parallel runs.

For the following examples, assume you have a project directory with multiple files and a directory structure:

```
project-directory
|-- images
|   |-- cat.png
|   +-- dog.png
|-- checkpoints
|   +-- model.h5
+-- model.h5
```

### Add a single file

The proceeding code snippet demonstrates how to add a single, local file to your artifact:

```python
# Add a single file
artifact.add_file(local_path="path/file.format")
```

For example, suppose you had a file called `'file.txt'` in your working local directory.

```python
artifact.add_file("path/file.txt")  # Added as `file.txt'
```

The artifact now has the following content:

```
file.txt
```

Optionally, pass the desired path within the artifact for the `name` parameter.

```python
artifact.add_file(local_path="path/file.format", name="new/path/file.format")
```

The artifact is stored as:

```
new/path/file.txt
```

| API Call                                                  | Resulting artifact |
| --------------------------------------------------------- | ------------------ |
| `artifact.add_file('model.h5')`                           | model.h5           |
| `artifact.add_file('checkpoints/model.h5')`               | model.h5           |
| `artifact.add_file('model.h5', name='models/mymodel.h5')` | models/mymodel.h5  |

### Add multiple files

The proceeding code snippet demonstrates how to add an entire, local directory to your artifact:

```python
# Recursively add a directory
artifact.add_dir(local_path="path/file.format", name="optional-prefix")
```

The proceeding API calls produce the proceeding artifact content:

| API Call                                    | Resulting artifact                                     |
| ------------------------------------------- | ------------------------------------------------------ |
| `artifact.add_dir('images')`                | <p><code>cat.png</code></p><p><code>dog.png</code></p> |
| `artifact.add_dir('images', name='images')` | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |
| `artifact.new_file('hello.txt')`            | `hello.txt`                                            |

### Add a URI reference

Artifacts track checksums and other information for reproducibility if the URI has a scheme that W&B library knows how to handle.

Add an external URI reference to an artifact with the [`add_reference`]({{< relref "/ref/python/sdk/classes/artifact.md#add_reference" >}}) method. Replace the `'uri'` string with your own URI. Optionally pass the desired path within the artifact for the name parameter.

```python
# Add a URI reference
artifact.add_reference(uri="uri", name="optional-name")
```

Artifacts currently support the following URI schemes:

* `http(s)://`: A path to a file accessible over HTTP. The artifact will track checksums in the form of etags and size metadata if the HTTP server supports the `ETag` and `Content-Length` response headers.
* `s3://`: A path to an object or object prefix in S3. The artifact will track checksums and versioning information (if the bucket has object versioning enabled) for the referenced objects. Object prefixes are expanded to include the objects under the prefix, up to a maximum of 10,000 objects.
* `gs://`: A path to an object or object prefix in GCS. The artifact will track checksums and versioning information (if the bucket has object versioning enabled) for the referenced objects. Object prefixes are expanded to include the objects under the prefix, up to a maximum of 10,000 objects.

The proceeding API calls will produce the proceeding artifacts:

| API call                                                                      | Resulting artifact contents                                          |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `artifact.add_reference('s3://my-bucket/model.h5')`                           | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/checkpoints/model.h5')`               | `model.h5`                                                           |
| `artifact.add_reference('s3://my-bucket/model.h5', name='models/mymodel.h5')` | `models/mymodel.h5`                                                  |
| `artifact.add_reference('s3://my-bucket/images')`                             | <p><code>cat.png</code></p><p><code>dog.png</code></p>               |
| `artifact.add_reference('s3://my-bucket/images', name='images')`              | <p><code>images/cat.png</code></p><p><code>images/dog.png</code></p> |

### Add files to artifacts from parallel runs

For large datasets or distributed training, multiple parallel runs might need to contribute to a single artifact.

```python
import wandb
import time

# We will use ray to launch our runs in parallel
# for demonstration purposes. You can orchestrate
# your parallel runs however you want.
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# Each batch of parallel writers should have its own
# unique group name.
group_name = "writer-group-{}".format(round(time.time()))


@ray.remote
def train(i):
    """
    Our writer job. Each writer will add one image to the artifact.
    """
    with wandb.init(group=group_name) as run:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)

        # Add data to a wandb table. In this case we use example data
        table = wandb.Table(columns=["a", "b", "c"], data=[[i, i * 2, 2**i]])

        # Add the table to folder in the artifact
        artifact.add(table, "{}/table_{}".format(parts_path, i))

        # Upserting the artifact creates or appends data to the artifact
        run.upsert_artifact(artifact)


# Launch your runs in parallel
result_ids = [train.remote(i) for i in range(num_parallel)]

# Join on all the writers to make sure their files have
# been added before finishing the artifact.
ray.get(result_ids)

# Once all the writers are finished, finish the artifact
# to mark it ready.
with wandb.init(group=group_name) as run:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # Create a "PartitionTable" pointing to the folder of tables
    # and add it to the artifact.
    artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)

    # Finish artifact finalizes the artifact, disallowing future "upserts"
    # to this version.
    run.finish_artifact(artifact)
```