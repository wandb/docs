---
description: Learn where W&B files are stored by default. Explore how to save, store
  sensitive information.
menu:
  default:
    identifier: data-privacy-and-compliance
    parent: artifacts
title: Artifact data privacy and compliance
---

Files are uploaded to Google Cloud bucket managed by W&B when you log artifacts. The contents of the bucket are encrypted both at rest and in transit. Artifact files are only visible to users who have access to the corresponding project.

{{< img src="/images/artifacts/data_and_privacy_compliance_1.png" alt="GCS W&B Client Server diagram" >}}

When you delete a version of an artifact, it is marked for soft deletion in our database and removed from your storage cost. When you delete an entire artifact, it is queued for permanently deletion and all of its contents are removed from the W&B bucket. If you have specific needs around file deletion please reach out to [Customer Support](mailto:support@wandb.com).

For sensitive datasets that cannot reside in a multi-tenant environment, you can use either a private W&B server connected to your cloud bucket or _reference artifacts_. Reference artifacts track references to private buckets without sending file contents to W&B. Reference artifacts maintain links to files on your buckets or servers. In other words, W&B only keeps track of the metadata associated with the files and not the files themselves.

{{< img src="/images/artifacts/data_and_privacy_compliance_2.png" alt="W&B Client Server Cloud diagram" >}}

Create a reference artifact similar to how you create a non reference artifact:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

For alternatives, contact us at [contact@wandb.com](mailto:contact@wandb.com) to talk about private cloud and on-premises installations.