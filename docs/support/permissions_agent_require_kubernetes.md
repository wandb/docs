---
title: "What permissions does the agent require in Kubernetes?"
displayed_sidebar: launch
---
The following Kubernetes manifest creates a role named `wandb-launch-agent` in the `wandb` namespace. This role allows the agent to create pods, configmaps, secrets, and access pod logs in the `wandb` namespace. The `wandb-cluster-role` enables the agent to create pods, access pod logs, create secrets, jobs, and check job status across any specified namespace.