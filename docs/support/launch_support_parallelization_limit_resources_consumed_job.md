---
title: "Does Launch support parallelization?  How can I limit the resources consumed by a job?"
displayed_sidebar: launch
---
Launch supports scaling jobs across multiple GPUs and nodes. Refer to [this guide](/tutorials/volcano) for details.

Each launch agent is configured with a `max_jobs` parameter, which determines the maximum number of simultaneous jobs it can run. Multiple agents can point to a single queue as long as they connect to an appropriate launching infrastructure.

You can set limits on CPU, GPU, memory, and other resources at the queue or job run level in the resource configuration. For information on setting up queues with resource limits on Kubernetes, refer to [this guide](../guides/launch/setup-launch-kubernetes).

For sweeps, include the following block in the queue configuration to limit the number of concurrent runs:

```yaml title="queue config"
  scheduler:
    num_workers: 4
```