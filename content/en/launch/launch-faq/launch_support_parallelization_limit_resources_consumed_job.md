---
menu:
  launch:
    identifier: launch_support_parallelization_limit_resources_consumed_job
    parent: launch-faq
title: Does Launch support parallelization?  How can I limit the resources consumed
  by a job?
---

Launch supports scaling jobs across multiple GPUs and nodes. Refer to the [Volcano integration guide]({{< relref "/launch/integration-guides/volcano.md" >}}) for details.

Each launch agent is configured with a `max_jobs` parameter, which determines the maximum number of simultaneous jobs it can run. Multiple agents can point to a single queue as long as they connect to an appropriate launching infrastructure.

You can set limits on CPU, GPU, memory, and other resources at the queue or job run level in the resource configuration. For information on setting up queues with resource limits on Kubernetes, see the [Kubernetes setup guide]({{< relref "/launch/set-up-launch/setup-launch-kubernetes.md" >}}).

For sweeps, include the following block in the queue configuration to limit the number of concurrent runs:

```yaml title="queue config"
  scheduler:
    num_workers: 4
```