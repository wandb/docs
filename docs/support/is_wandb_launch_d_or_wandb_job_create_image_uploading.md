---
title: "Is `wandb launch -d` or `wandb job create image` uploading a whole docker artifact and not pulling from a registry?"
tags:
   - launch
---

No. The  `wandb launch -d` command will not upload to a registry for you. You need to upload your image to a registry yourself. Here are the general steps:

1. Build an image. 
2. Push the image to a registry.

The workflow looks like:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

From there, the launch agent will spin up a job pointing to that container.  See [Advanced agent setup](./setup-agent-advanced.md#agent-configuration) for examples of how to give the agent access to pull an image from a container registry.

For Kubernetes, the Kubernetes cluster pods will need access to the registry you are pushing to.