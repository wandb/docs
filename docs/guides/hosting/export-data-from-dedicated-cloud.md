---
description: Export data from Dedicated Cloud
displayed_sidebar: default
title: Export data from Dedicated Cloud
---

If you would like to export all the data managed in your Dedicated Cloud instance, you can use the W&B SDK API to extract the runs, metrics, artifacts, and more with the [Import and Export API](/ref/python/public-api/). The following table has covers some of the key exporting use cases.

| Purpose | Documentation |
|---------|---------------|
| Export project metadata | [Projects API](/ref/python/public-api/projects/) |
| Export runs in a project | [Runs API](/ref/python/public-api/runs/)) |
| Export reports | [Reports API](/guides/reports/clone-and-export-reports/) |
| Export artifacts | [Explore artifact graphs](/guides/artifacts/explore-and-traverse-an-artifact-graph), [Download and use artifacts](/guides/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb) |

:::info
You manage artifacts stored in the Dedicated Cloud with [Secure Storage Connector](/guides/app/features/teams/#secure-storage-connector). In that case, you may not need to export the artifacts using the W&B SDK API.
:::

:::note
Using W&B SDK API to export all of your data can be slow if you have a large number of runs, artifacts etc. W&B recommends running the export process in appropriately sized batches so as not to overwhelm your Dedicated Cloud instance.
:::