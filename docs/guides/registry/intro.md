---
slug: /guides/registry
displayed_sidebar: default
---

# Registry

:::info
W&B Registry is in active development.  
:::


Use W&B Registry to share artifacts from your machine learning pipeline, such as machine learning and datasets, across teams within an organization. 


## How it works
W&B Registry is composed of three main components: registries, collections, and [artifact versions](../artifacts/create-a-new-artifact-version.md).

A *registry* is a [INSERT]. You can think of a registry as the top most level of a directory. Each registry consists of one or more sub directories called collections. A *collection* is a folder or a set of linked [*artifact versions*](../artifacts/create-a-new-artifact-version.md) inside a registry. An [artifact version](../artifacts/create-a-new-artifact-version.md) is [INSERT].  A registry belongs to an organization, not a specific team.

Consider the following example, suppose your company wants to explore brand new cat classification models within your organization. To do this, you create a registry called "Cat Models registry". You are not sure which image classification is best, so you create multiple machine learning experiments with varying hyperparameters and algorithms. To organize your experiments and results, you create a collection for each model algorithm that experiment. Within that collection, you link the best model artifacts from your experiments to that collection.

<!-- To do: Add Raven's new diagram -->

## Migrating from W&B Model Registry to W&B Registry

[INSERT blurb on how this impacts customers]

 

<!-- To do: INSERT -->