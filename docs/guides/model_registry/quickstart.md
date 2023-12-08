---
description: Try the model registry in one minute. Learn how to log a model version, then link it to a registered model.
displayed_sidebar: default
---
# Quickstart

Try the model registry in one minute. This quickstart will show you how to log a model version and link it to a registered model. Visit [this](https://docs.wandb.ai/guides/track/log/log-models#log-and-link-a-model-to-the-wb-model-registry) guide to learn more about the model APIs used below. 

Copy and paste the following code sample into your Jupyter Notebook or in a Python script.

```python
import wandb
import random

# Start a new W&B run
with wandb.init(project="models_quickstart") as run:
    # Simulate logging model metrics
    run.log({"acc": random.random()})

    # Create a simulated model file
    with open("my_model.h5", "w") as f:
        f.write("Model: " + str(random.random()))

    # Save the model file to W&B and link the model to the Model Registry
    run.link_model(path="my_model.h5", registered-model-name="Bookmarked-Model-Checkpoints")

    run.finish()
```
