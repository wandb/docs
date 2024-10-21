---
title: "Does Launch support parallelization?  How can I limit the resources consumed by a job?"
tags:
   - launch
---

Yes, Launch supports scaling jobs across mulitple GPUs and multiple nodes.  See [this guide](/tutorials/volcano) for details.

On an inter-job level, an individual launch agent is configured with a `max_jobs` parameter that determines how many jobs that agent can run simultaneously. Additionally, you can point to as many agents as you want at a particular queue, so long as those agents are connected to an infrastructure that they can launch into.
  
You can limit the CPU/GPU, memory, and other requirements at the launch queue or job run level, in the resource config. For more information about setting up queues with resource limits on Kubernetes see [here](setup-launch-kubernetes). 

For sweeps, in the SDK you can add a block to the queue config

```yaml title="queue config"
  scheduler:
    num_workers: 4
```
To limit the number of concurrent runs from a sweep that will be run in parallel.