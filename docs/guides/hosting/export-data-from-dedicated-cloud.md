---
description: Export data from Dedicated Cloud
displayed_sidebar: default
---

# Export data from Dedicated Cloud

If you would like to export all the data managed in your Dedicated Cloud instance, you may use the Wandb SDK API to extract the runs, metrics, artifacts etc. and log those to another cloud or on-premises storage using APIs relevant to that storage. 

Some use cases for data export are mentioned at [Import and Export Data](../track/public-api-guide.md#export-data). Another use case would be if you are planning to end your agreement to use Dedicated Cloud, you may want to export the pertinent data before W&B terminates the instance.

Refer to the table below for APIs and other documentation:

| Purpose | Documentation |
|---------|---------------|
| Export project metadata | [Projects API](../../ref/python/public-api/api.md#projects) |
| Export runs in a project | [Runs API](../../ref/python/public-api/api.md#runs), [Export run data](../track/public-api-guide.md#export-run-data), [Querying multiple runs](../track/public-api-guide.md#querying-multiple-runs) |
| Export reports | [Reports API](../../ref/python/public-api/api.md#reports) |
| Export artifacts | [Artifact API](../../ref/python/public-api/api.md#artifact), [Explore and traverse an artifact graph](../artifacts/explore-and-traverse-an-artifact-graph.md#traverse-an-artifact-programmatically), [Download and use an artifact](../artifacts/download-and-use-an-artifact.md#download-and-use-an-artifact-stored-on-wb) |

:::info
When using Dedicated Cloud with [Secure Storage Connector](./secure-storage-connector.md), your artifacts are in a storage that is managed by you. In that case, you may not need to export the artifacts using the Wandb SDK API.
:::

:::note
Using Wandb SDK APIs to export all of your data can be slow if you have a large number of runs, artifacts etc. W&B recommends running the export process in appropriately sized batches so as not to overwhelm your Dedicated Cloud instance.
:::