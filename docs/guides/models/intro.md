---
slug: /guides/models
description: Model registry to manage the model lifecycle from training to production
displayed_sidebar: default
---

# Models 

Use W&B Models as a central system of record for your best models, standardized and organized in a model registry across projects and teams. 

With W&B Models, you can:

* [Bookmark your best model versions for each machine learning task.](./model_tags.md)
* Move model versions through its ML lifecycle; from staging to production.
* Track and audit the history of changes to production models.




![](/images/models/models_landing_page.png)

## How it works
Track and manage your trained models with a few simple steps.

1. **Log a model version**: In your training script, add a few lines of code to save the model files as an artifact to W&B. 
2. **Compare performance**: Check live charts to compare the metrics and sample predictions from model training and validation. Identify which model version performed the best.
3. **Link to registry**: Bookmark the best model version by linking it to a registered model, either programmatically in Python or interactively in the W&B UI.

The following code snippet demonstrates how to log and link a model to the Model Registry:

```python showLineNumbers
import wandb
import random

# Start a new W&B run
with wandb.init(project="models_quickstart") as run:

  # Simulate logging model metrics
  run.log({"acc": random.random()})

  # Create a simulated model file
  with open("my_model.h5", "w") as f: f.write("Model: " + str(random.random()))

  # Save the dummy model to W&B
  best_model = wandb.Artifact(f"model_{run.id}", type='model')
  best_model.add_file('my_model.h5')
  run.log_artifact(best_model)

  # Link the model to the Model Registry
  run.link_artifact(best_model, 'model-registry/My Registered Model')

  run.finish()
```

4. **Test and deploy your model**: Transition model versions through customizable workflows stages, such as `staging` and `production`.

## How to get started
Depending on your use case, explore the following resources to get started with W&B Models:

<!-- * [Try the Quickstart](./quickstart.md) to log and link a sample model in just two minutes. -->
* Check out the [Model Registry Quickstart YouTube](https://www.youtube.com/watch?v=jy9Pk9riwZI&ab_channel=Weights%26Biases) video.
* Read the [models walkthrough](./walkthrough.md) for a step-by-step outline of the W&B Python SDK commands you could use to create, track, and use a dataset artifact.
* Explore this chapter to learn how about:
   * [Role based access controls (RBAC)](./access_controls.md).
   * [How to automate model testing and deployment](./automation.md).
   * [Use tags to organize registered models](./model_tags.md).
   * Set up [Slack notifications](./notifications.md) when a new model version is linked to a model registry.


