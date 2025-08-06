---
menu:
  default:
    identifier: ko-guides-models-track-runs-manage-runs
    parent: what-are-runs
title: Move runs
---

This page shows how to move a run from one project to another, into or out of a team, or from one team to another. You must have access to the run at its current and new locations.

{{% alert %}}
When you move a run, historical artifacts associated with it are not moved. To move an artifact manually, you can use the [`wandb artifact get`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-get/" lang="ko" >}}) SDK command or the [`Api.artifact` API]({{< relref path="/ref/python/public-api/api/#artifact" lang="ko" >}}) to download the artifact, then use [`wandb artifact put`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put/" lang="ko" >}}) or the `Api.artifact` API to upload it to the run's new location.
{{% /alert %}}

To customize the **Runs** tab, refer to [Project page]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ko" >}}).

If you group runs into experiments, refer to [Set a group in the UI]({{< relref path="grouping.md#set-a-group-in-the-ui" lang="ko" >}}).

## Move runs between your projects

To move runs from one project to another:

1. Navigate to the project that contains the runs you want to move.
2. Select the **Runs** tab from the project sidebar.
3. Select the checkbox next to the runs you want to move.
4. Choose the **Move** button above the table.
5. Select the destination project from the dropdown.

{{< img src="/images/app_ui/howto_move_runs.gif" alt="Demo of moving a run between projects" >}}

## Move runs to a team

Move runs to a team you are a member of:

1. Navigate to the project that contains the runs you want to move.
2. Select the **Runs** tab from the project sidebar.
3. Select the checkbox next to the runs you want to move.
4. Choose the **Move** button above the table.
5. Select the destination team and project from the dropdown.

{{< img src="/images/app_ui/demo_move_runs.gif" alt="Demo of moving a run to a team" >}}