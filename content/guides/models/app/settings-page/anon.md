---
description: Log and visualize data without a W&B account
menu:
  default:
    identifier: anon
    parent: settings
title: Anonymous mode
weight: 80
---

Are you publishing code that you want anyone to be able to run easily? Use anonymous mode to let someone run your code, see a W&B dashboard, and visualize results without needing to create a W&B account first.

Allow results to be logged in anonymous mode with: 

```python
import wandb

wandb.init(anonymous="allow")
```

For example, the proceeding code snippet shows how to create and log an artifact with W&B:

```python
import wandb

run = wandb.init(anonymous="allow")

artifact = wandb.Artifact(name="art1", type="foo")
artifact.add_file(local_path="path/to/file")
run.log_artifact(artifact)

run.finish()
```

[Try the example notebook](https://colab.research.google.com/drive/1nQ3n8GD6pO-ySdLlQXgbz4wA3yXoSI7i) to see how anonymous mode works.