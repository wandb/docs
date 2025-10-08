---
title: "Prerequisites"
linkTitle: "Prerequisites"
weight: 1
description: >
  Set up your environment to use W&B Training.
---

Complete these steps before using W&B Training features through the OpenPipe ART framework or API.

{{< alert title="Tip" >}}
Before starting, review the [usage information and limits]({{< relref "guides/training/serverless-rl/usage-limits" >}}) to understand costs and restrictions.
{{< /alert >}}

## Sign up and create an API key

To authenticate your machine with W&B, you must first generate an API key at [wandb.ai/authorize](https://wandb.ai/authorize). Copy the API key and store it securely.

## Create a project in W&B

Create a project in your W&B account to track usage, record training metrics, and save trained models. See the [Projects guide]({{< relref "/guides/track/project-page" >}}) for more information.

## Next steps

After completing the prerequisites:

* Check the [API reference]({{< relref "/ref/training" >}}) to learn about available endpoints
* Try the [ART quickstart](https://art.openpipe.ai/getting-started/quick-start)
