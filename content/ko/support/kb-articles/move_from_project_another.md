---
menu:
  support:
    identifier: ko-support-kb-articles-move_from_project_another
support:
- runs
title: Is it possible to move a run from one project to another?
toc_hide: true
type: docs
url: /support/:filename
---

You can move a run from one project to another by following these steps:

- Navigate to the project page with the run to be moved.
- Click on the **Runs** tab to open the runs table.
- Select the runs to move.
- Click the **Move** button.
- Choose the destination project and confirm the action.

W&B supports moving runs through the UI, but does not support copying runs. Artifacts logged with the runs do not transfer to the new project. To move artifacts to the run's new location manually, you can use the [`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ko" >}}) SDK command or the [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ko" >}}) to download the artifact, then use [wandb artifact put]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ko" >}}) or the `Api.artifact` API to upload it to the run's new location.