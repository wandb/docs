---
description: Try the model registry in one minute. Learn how to log a model version, then link it to a registered model.
displayed_sidebar: default
---
# Quickstart

Try the model registry in one minute. This quickstart will show you how to log a model version and link to a registered model.

Copy and paste the following code sample into your Jupyter Notebook or in a Python script.

```python
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