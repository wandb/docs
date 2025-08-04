---
description: Export data from Dedicated cloud
menu:
  default:
    identifier: export-data-from-dedicated-cloud
    parent: dedicated-cloud
title: Export data from Dedicated cloud
url: guides/hosting/export-data-from-dedicated-cloud
---

If you would like to export all the data managed in your Dedicated cloud instance, you can use the W&B SDK API to extract the runs, metrics, artifacts, and more with the [Import and Export API]({{< relref "/ref/python/public-api/index.md" >}}). The following table has covers some of the key exporting use cases.

| Purpose | Documentation |
|---------|---------------|
| Export project metadata | [Projects API]({{< relref "/ref/python/public-api/projects.md" >}}) |
| Export runs in a project | [Runs API]({{< relref "/ref/python/public-api/runs.md" >}}) |
| Export reports | [Report and Workspace API]({{< relref "/guides/core/reports/clone-and-export-reports/" >}}) |
| Export artifacts | [Explore artifact graphs]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph" >}}), [Download and use artifacts]({{< relref "/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" >}}) |

If you manage artifacts stored in the Dedicated cloud with [Secure Storage Connector]({{< relref "/guides/models/app/features/teams.md#secure-storage-connector" >}}), you may not need to export the artifacts using the W&B SDK API.

{{% alert %}}
Using W&B SDK API to export all of your data can be slow if you have a large number of runs, artifacts etc. W&B recommends running the export process in appropriately sized batches so as not to overwhelm your Dedicated cloud instance.
{{% /alert %}}