---
title: Reports and Workspaces API
no_list: true
weight: 9
---

The W&B Reports and Workspaces API, accessible at `wandb_workspaces`, allows you to create [reports]({{< relref "/guides/core/reports" >}}), which can be published on the web to share findings, and well as customize a [workspace]({{< relref "/guides/models/app/features/cascade-settings" >}}) where training and fine-tuning work was done. 


{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/" >}}

{{% alert %}}
W&B Report and Workspace API is in Public Preview.
{{% /alert %}}

## Installation and setup

### Sign up and create an API key

To authenticate your machine with W&B, you must first generate an API key at https://wandb.ai/authorize.

### Install and import packages

Install the W&B Report and Workspaces library.

```
pip install wandb-workspaces
```

### Import W&B Python SDK:

```python
import wandb_workspaces
```

Specify the entity of your team in the following code block:

```python
TEAM_ENTITY = "<Team_Entity>" # Replace with your team entity
PROJECT = "my-awesome-project"
```

## Classes

[`reports`](./reports.md)

[`workspaces`](./workspaces.md)
