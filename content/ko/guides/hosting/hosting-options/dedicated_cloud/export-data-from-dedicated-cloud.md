---
description: Export data from Dedicated cloud
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-dedicated_cloud-export-data-from-dedicated-cloud
    parent: dedicated-cloud
title: Export data from Dedicated cloud
url: guides/hosting/export-data-from-dedicated-cloud
---

If you would like to export all the data managed in your Dedicated cloud instance, you can use the W&B SDK API to extract the runs, metrics, artifacts, and more with the [Import and Export API]({{< relref path="/ref/python/public-api/" lang="ko" >}}). The following table has covers some of the key exporting use cases.

| Purpose | Documentation |
|---------|---------------|
| Export project metadata | [Projects API]({{< relref path="/ref/python/public-api/projects/" lang="ko" >}}) |
| Export runs in a project | [Runs API]({{< relref path="/ref/python/public-api/runs/" lang="ko" >}}) |
| Export reports | [Reports API]({{< relref path="/guides/core/reports/clone-and-export-reports/" lang="ko" >}}) |
| Export artifacts | [Explore artifact graphs]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ko" >}}), [Download and use artifacts]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" lang="ko" >}}) |

If you manage artifacts stored in the Dedicated cloud with [Secure Storage Connector]({{< relref path="/guides/models/app/settings-page/teams/#secure-storage-connector" lang="ko" >}}), you may not need to export the artifacts using the W&B SDK API.

{{% alert %}}
Using W&B SDK API to export all of your data can be slow if you have a large number of runs, artifacts etc. W&B recommends running the export process in appropriately sized batches so as not to overwhelm your Dedicated cloud instance.
{{% /alert %}}