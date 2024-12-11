---
menu:
  launch:
    identifier: launch_d_wandb_job_create_image_uploading_whole_docker
    parent: launch-faq
title: Is `wandb launch -d` or `wandb job create image` uploading a whole docker artifact
  and not pulling from a registry?
---

No, the `wandb launch -d` command does not upload images to a registry. Upload images to a registry separately. Follow these steps:

1. Build an image.
2. Push the image to a registry.

The workflow is as follows:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

The launch agent then spins up a job pointing to the specified container. See [Advanced agent setup](../guides/launch/setup-agent-advanced.md#agent-configuration) for examples on configuring agent access to pull images from a container registry.

For Kubernetes, ensure that the Kubernetes cluster pods have access to the registry where the image is pushed.