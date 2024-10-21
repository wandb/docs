---
title: "Can Launch automatically provision (and spin down) compute resources for me in the target environment?"
tags:
   - launch
---

This depends on the environment, we are able to provision resources in SageMaker, and Vertex. In Kubernetes, autoscalers can be used to automatically spin up and spin down resources when required. The Solution Architects at W&B are happy to work with you to configure your underlying Kubernetes infrastructure to facilitate retries, autoscaling, and use of spot instance node pools. Reach out to support@wandb.com or your shared Slack channel.