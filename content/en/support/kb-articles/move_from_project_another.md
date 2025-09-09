---
url: /support/:filename
title: Is it possible to move a run from one project to another?
toc_hide: true
type: docs
support:
- runs
---
You can move a run from one project to another by following these steps:

- Navigate to the project page with the run to be moved.
- Click on the **Runs** tab to open the runs table.
- Select the runs to move.
- Click the **Move** button.
- Choose the destination project and confirm the action.

W&B supports moving runs through the UI, but does not support copying runs. Artifacts logged with the runs do not transfer to the new project. To move artifacts to the run's new location manually, you can use the [`wandb artifact get`]({{< relref "/ref/cli/wandb-artifact/wandb-artifact-get/" >}}) SDK command or the [`Api.artifact` API]({{< relref "/ref/python/sdk/public-api/api/#artifact" >}}) to download the artifact, then use [wandb artifact put]({{< relref "/ref/cli/wandb-artifact/wandb-artifact-put/" >}}) or the `Api.artifact` API to upload it to the run's new location.