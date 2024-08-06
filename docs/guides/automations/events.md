---
displayed_sidebar: default
---

An *event* is a change that takes place in the W&B ecosystem. 

## Registry Event Types

Registries support two event types: **Linking a new artifact to a registered model** and **Adding a new alias to a version of the registered model**.


### Link a model version
Use the **Linking a new artifact to a registered model** event type to test new model candidates. 

### Create a custom alias
Use the **Adding a new alias to a version of the registered model** event type to specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied.


See [Link a model version](../model_registry/link-model-version.md) for information on how to link model versions and [Create a custom alias](../artifacts/create-a-custom-alias.md) for information an artifact aliases.


## Artifacts Event Types
You can define two different event types for artifact collections in your project: **A new version of an artifact is created in a collection** and **An artifact alias is added**.

### Artifact is created in a collection
Use the **A new version of an artifact is created in a collection** event type for applying recurring actions to each version of an artifact. For example, you can create an automation that automatically starts a training job when a new dataset artifact version is created.

### An artifact alias is added
Use the **An artifact alias is added** event type to create an automation that activates when a specific alias is applied to an artifact version. For example, you could create an automation that triggers an action when someone adds "test-set-quality-check" alias to an artifact that then triggers downstream processing on that dataset. 