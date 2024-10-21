---
title: "I do not want W&B to build a container for me, can I still use Launch?"
tags: []
---

### I do not want W&B to build a container for me, can I still use Launch?
Yes. Run the following to launch a pre-built docker image. Replace the items in the `<>` with your information:

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```  

This will build a job when you create a run.

Or you can make a job from an image:

```bash
wandb job create image <image-name> -p <project> -e <entity>
```