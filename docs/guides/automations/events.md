---
displayed_sidebar: default
---

An *event* is a change that takes place in the W&B ecosystem. 

## Registry event types

Registries support two event types: **Linking a new artifact to a registered model** and **Adding a new alias to a version of the registered model**.


### Link a model version

Use the **Linking a new artifact to a registered model** event type to test new model candidates. For example, if you ran this code, a configured action would trigger:

```python
import wandb
run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

### Create a custom alias

Use the **Adding a new alias to a version of the registered model** event type to specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied. For example, if you ran this code, a configured action would trigger:

```python
artifact = wandb.Artifact(name= "test-model", type="model")
artifact.add_file("model.h5")
run.log_artifact(artifact, aliases=["latest", "deploy"])
```

See [Link a model version](../model_registry/link-model-version.md) for information on how to link model versions and [Create a custom alias](../artifacts/create-a-custom-alias.md) for information an artifact aliases.


## Artifacts event types

You can define two different event types for artifact collections in your project: **A new version of an artifact is created in a collection** and **An artifact alias is added**.

### Artifact is created in a collection

Use the **A new version of an artifact is created in a collection** event type for applying recurring actions to each version of an artifact. For example, you can create an automation that automatically starts a training job after creating a new dataset artifact version. For example, if you ran this code, it would trigger a configured action (assuming you already have an active collection):

```python
with wandb.init() as run:
    artifact = wandb.Artifact(name= "dataset-collection", type="dataset")
    artifact.add_file("dataset2.csv")
    run.upsert_artifact(artifact, distributed_id="my_dist_artifact")
```

### An artifact alias is added

Use the **An artifact alias is added** event type to create an automation that activates when an artifact version receives a specific alias . For example, you could create an automation that triggers an action when someone adds "test-set-quality-check" alias to an artifact that then triggers downstream processing on that dataset. For example, if you ran this code, a configured action would trigger:

```python
artifact = wandb.Artifact(name= "test-dataset", type="dataset")
artifact.add_file("test-data.csv")
run.log_artifact(artifact, aliases=["latest", "test-set-quality-check"])
```