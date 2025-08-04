---
description: Track files saved in an external bucket, HTTP file server, or an NFS share.
menu:
  default:
    identifier: track-external-files
    parent: artifacts
title: Track external files
weight: 7
---

Use **reference artifacts** to track and use files saved outside of W&B servers, for example in CoreWeave AI Object Storage, an Amazon Simple Storage Service (Amazon S3) bucket, GCS bucket, Azure blob, HTTP file server, or NFS share.

W&B logs metadata about the the object, such as the object's ETag and size. If object versioning is enabled on the bucket, the version ID is also logged.

{{% alert %}}
If you log an artifact that does not track external files, W&B saves the artifact's files to W&B servers. This is the default behavior when you log artifacts with the W&B Python SDK.

See the [Artifacts quickstart]({{< relref "/guides/core/artifacts/artifacts-walkthrough" >}}) for information on how to save files and directories to W&B servers instead.
{{% /alert %}}

The following describes how to construct reference artifacts.

## Track an artifact in an external bucket

Use the W&B Python SDK to track references to files stored outside of W&B.

1. Initialize a run with `wandb.init()`.
2. Create an artifact object with `wandb.Artifact()`.
3. Specify the reference to the bucket path with the artifact object's `add_reference()` method.
4. Log the artifact's metadata with `run.log_artifact()`.

```python
import wandb

# Initialize a W&B run
run = wandb.init()

# Create an artifact object
artifact = wandb.Artifact(name="name", type="type")

# Add a reference to the bucket path
artifact.add_reference(uri = "uri/to/your/bucket/path")

# Log the artifact's metadata
run.log_artifact(artifact)
run.finish()
```

Suppose your bucket has the following directory structure:

```text
s3://my-bucket

|datasets/
  |---- mnist/
|models/
  |---- cnn/
```

The `datasets/mnist/` directory contains a collection of images. Track the directory as a dataset with `wandb.Artifact.add_reference()`. The following code sample creates a reference artifact `mnist:latest` using the artifact object's `add_reference()` method.

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact(name="mnist", type="dataset")
artifact.add_reference(uri="s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
run.finish()
```

Within the W&B App, you can look through the contents of the reference artifact using the file browser, [explore the full dependency graph]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph" >}}), and scan through the versioned history of your artifact. The W&B App does not render rich media such as images, audio, and so forth because the data itself is not contained within the artifact.

{{% alert %}}
W&B Artifacts support any Amazon S3 compatible interface, including MinIO. The scripts described below work as-is with MinIO, when you set the `AWS_S3_ENDPOINT_URL` environment variable to point at your MinIO server.
{{% /alert %}}

{{% alert color="secondary" %}}
By default, W&B imposes a 10,000 object limit when adding an object prefix. You can adjust this limit by specifying `max_objects=` when you call `add_reference()`.
{{% /alert %}}

## Download an artifact from an external bucket

W&B  retrieves the files from the underlying bucket when it downloads a reference artifact using the metadata recorded when the artifact is logged. If your bucket has object versioning enabled, W&B retrieves the object version that corresponds to the state of the file at the time an artifact was logged. As you evolve the contents of your bucket, you can always point to the exact version of your data a given model was trained on,  because the artifact serves as a snapshot of your bucket during the training run.

The following code sample shows how to download a reference artifact. The the APIs for downloading artifacts are the same for both reference and non-reference artifacts:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

{{% alert %}}
W&B recommends that you enable 'Object Versioning' on your storage buckets if you overwrite files as part of your workflow. With versioning enabled on your buckets, artifacts with references to files that have been overwritten will still be intact because the older object versions are retained. 

Based on your use case, read the instructions to enable object versioning: [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/using-object-versioning#set), [Azure](https://learn.microsoft.com/azure/storage/blobs/versioning-enable).
{{% /alert %}}

### Add and download an external reference example

The following code sample uploads a dataset to an Amazon S3 bucket, tracks it with a reference artifact, then downloads it:

```python
import boto3
import wandb

run = wandb.init()

# Training here...

s3_client = boto3.client("s3")
s3_client.upload_file(file_name="my_model.h5", bucket="my-bucket", object_name="models/cnn/my_model.h5")

# Log the model artifact
model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

At a later point, you can download the model artifact. Specify the name of the artifact and its type:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact(artifact_or_name = "cnn", type="model")
datadir = artifact.download()
```

{{% alert %}}
See the following reports for an end-to-end walkthrough on how to track artifacts by reference for GCP or Azure:

* [Guide to Tracking Artifacts by Reference with GCP](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Working with Reference Artifacts in Microsoft Azure](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}

## Cloud storage credentials

W&B uses the default mechanism to look for credentials based on the cloud provider you use. Read the documentation from your cloud provider to learn more about the credentials used:

| Cloud provider | Credentials Documentation |
| -------------- | ------------------------- |
| CoreWeave AI Object Storage | [CoreWeave AI Object Storage documentation](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/manage-access-keys/cloud-console-tokens) |
| AWS            | [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure documentation](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

For AWS, if the bucket is not located in the configured user's default region, you must set the `AWS_REGION` environment variable to match the bucket region.

{{% alert color="secondary" %}}
Rich media such as images, audio, video, and point clouds may fail to render in the App UI depending on the CORS configuration of your bucket. Allow listing **app.wandb.ai** in your bucket's CORS settings will allow the App UI to properly render such rich media.

If rich media such as images, audio, video, and point clouds does not render in the App UI, ensure that `app.wandb.ai` is allowlisted in your bucket's CORS policy.
{{% /alert %}}

## Track an artifact in a filesystem

Another common pattern for fast access to datasets is to expose an NFS mount point to a remote filesystem on all machines running training jobs. This can be an even simpler solution than a cloud storage bucket because from the perspective of the training script, the files look just like they are sitting on your local filesystem. Luckily, that ease of use extends into using Artifacts to track references to file systems, whether they are mounted or not.

Suppose you have a filesystem mounted at `/mount` with the following structure:

```bash
mount
|datasets/
		|-- mnist/
|models/
		|-- cnn/
```

Within `mnist/` is a dataset, a collection of images. You can track it with an artifact:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```
{{% alert color="secondary" %}}
By default, W&B imposes a 10,000 file limit when adding a reference to a directory. You can adjust this limit by specifying `max_objects=` when you call `add_reference()`.
{{% /alert %}}

Note the triple slash in the URL. The first component is the `file://` prefix that denotes the use of filesystem references. The second component begins the path to the dataset, `/mount/datasets/mnist/`.

The resulting artifact `mnist:latest` looks and acts like a regular artifact. The only difference is that the artifact only consists of metadata about the files, such as their sizes and MD5 checksums. The files themselves never leave your system.

You can interact with this artifact just as you would a normal artifact. In the UI, you can browse the contents of the reference artifact using the file browser, explore the full dependency graph, and scan through the versioned history of your artifact. However, the UI cannot render rich media such as images, audio, because the data itself is not contained within the artifact.

Downloading a reference artifact:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

For a filesystem reference, a `download()` operation copies the files from the referenced paths to construct the artifact directory. In the above example, the contents of `/mount/datasets/mnist` are copied into the directory `artifacts/mnist:v0/`. If an artifact contains a reference to a file that was overwritten, then `download()` will throw an error because the artifact can no longer be reconstructed.

Putting it all together, you can use the following code to track a dataset under a mounted filesystem that feeds into a training job:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# Track the artifact and mark it as an input to
# this run in one swoop. A new artifact version
# is only logged if the files under the directory
# changed.
run.use_artifact(artifact)

artifact_dir = artifact.download()

# Perform training here...
```

To track a model, log the model artifact after the training script writes the model files to the mount point:

```python
import wandb

run = wandb.init()

# Training here...

# Write model to disk

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```


<!-- ### Log artifacts outside of runs

W&B creates a run when you log an artifact outside of a run. Each artifact belongs to a run, which in turn belongs to a project. An artifact (version) also belongs to a collection, and has a type.

Use the [`wandb artifact put`]({{< relref "/ref/cli/wandb-artifact/wandb-artifact-put" >}}) command to upload an artifact to the W&B server outside of a W&B run. Provide the name of the project you want the artifact to belong to along with the name of the artifact (`project/artifact_name`).Optionally provide the type (`TYPE`). Replace `PATH` in the code snippet below with the file path of the artifact you want to upload.

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

W&B will create a new project if a the project you specify does not exist. For information on how to download an artifact, see [Download and use artifacts]({{< relref "/guides/core/artifacts/download-and-use-an-artifact" >}}). -->