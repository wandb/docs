---
menu:
  launch:
    identifier: build_container_launch
    parent: launch-faq
title: I do not want W&B to build a container for me, can I still use Launch?
---

To launch a pre-built Docker image, execute the following command. Replace the placeholders in the `<>` with your specific information:

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

This command creates a job and starts a run.

To create a job from an image, use the following command:

```bash
wandb job create image <image-name> -p <project> -e <entity>
```