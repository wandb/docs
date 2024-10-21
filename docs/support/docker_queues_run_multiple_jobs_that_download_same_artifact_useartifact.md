---
title: "When using Docker queues to run multiple jobs that download the same artifact with `use_artifact`, do we re-download the artifact for every single run of the job, or is there any caching going on under the hood?"
tags:
   - launch
---

There is no caching; each job is independent.  However, there are ways to configure your queue/agent where it mounts a shared cache.  You can achieve this via docker args in the queue config.

As a special case, you can also mount the W&B artifacts cache as a persistent volume.