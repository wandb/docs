---
description: Discover how to create a job for W&B Launch.
---

# Create a job

## What is a job?

A job is a complete blueprint of how to perform a step in your ML workflow, like training a model, running an evaluation, or deploying a model to an inference server. For more information, see the [details of launched jobs section](launch-jobs#view-details-of-launched-jobs).

## How do I create a job?

Jobs will be captured automatically from any workloads that you track with W&B if your run has associated source code. You can connect source code to your runs in the following ways:

1. [Log your code as an Artifact](../app/features/panels/code.md#save-library-code).
2. Associate your run with a [git commit](../../guides/track/tracking-faq.md#how-can-i-save-the-git-commit-associated-with-my-run).
3. Set the [`WANDB_DOCKER` environment variable](../../guides/integrations/other/docker.md) to capture a container image as a source for your job.

:::info
You must `wandb>=0.13.8` in order to create jobs from your runs.
:::

## Job naming conventions

By default, W&B automatically creates a job name for you. The name is generated depending on how the job is created (GitHub, code artifact, or Docker image). The following table describes the job naming convention used for each job source:

| Source        | Naming convention                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |
