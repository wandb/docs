---
title: Python SDK
module: 
weight: 2
---
The W&B Python SDK, accessible at `wandb`, enables you to train and fine-tune models, and manage models from experimentation to production. 

> After performing your training and fine-tuning operations with this SDK, you can use [the Public API]({{< relref "/ref/python/sdk/public-api" >}}) to query and analyze the data that was logged, and [the Reports and Workspaces API]({{< relref "/ref/python/wandb_workspaces" >}}) to generate a web-publishable [report]({{< relref "/guides/core/reports" >}}) summarizing your work.

## Installation and setup

### Sign up and create an API key

To authenticate your machine with W&B, you must first generate an API key at https://wandb.ai/authorize.

### Install and import packages

Install the W&B library.

```
pip install wandb
```

### Import W&B Python SDK:

```python
import wandb
```

Specify the entity of your team in the following code block:

```python
TEAM_ENTITY = "<Team_Entity>" # Replace with your team entity
PROJECT = "my-awesome-project"
```
