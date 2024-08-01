---
displayed_sidebar: default
---

An *event* is a change that takes place in the W&B ecosystem. 

## Model Registry

The Model Registry supports two event types: **Linking a new artifact to a registered model** and **Adding a new alias to a version of the registered model**.

See [Link a model version](./link-model-version.md) for information on how to link model versions and [Create a custom alias](../artifacts/create-a-custom-alias.md) for information an artifact aliases.



:::tip
Use the **Linking a new artifact to a registered model** event type to test new model candidates. Use the **Adding a new alias to a version of the registered model** event type to specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied.
:::

## Artifacts
You can define two different event types for artifact collections in your project: **A new version of an artifact is created in a collection** and **An artifact alias is added**.

:::tip
Use the **A new version of an artifact is created in a collection** event type for applying recurring actions to each version of an artifact. For example, you can create an automation that automatically starts a training job when a new dataset artifact version is created.

Use the **An artifact alias is added** event type to create an automation that activates when a specific alias is applied to an artifact version. For example, you could create an automation that triggers an action when someone adds "test-set-quality-check" alias to an artifact that then triggers downstream processing on that dataset. 