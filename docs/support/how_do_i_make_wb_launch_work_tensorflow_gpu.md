---
title: "How do I make W&B Launch work with Tensorflow on GPU?"
tags: []
---

### How do I make W&B Launch work with Tensorflow on GPU?
For jobs that use tensorflow on GPU, you may also need to specify a custom base image for the container build that the agent will perform in order for your runs to properly utilize GPUs. This can be done by adding an image tag under the `builder.accelerator.base_image` key to the resource configuration. For example:

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```

Note prior to wandb version: 0.15.6 use `cuda` instead of `accelerator` as the parent key to `base_image`.