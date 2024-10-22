---
title: "When using Docker queues to run multiple jobs that download the same artifact with `use_artifact`, do we re-download the artifact for every single run of the job, or is there any caching going on under the hood?"
tags:
   - launch
---
No caching exists; each job operates independently. Configure the queue or agent to mount a shared cache using Docker arguments in the queue configuration.

Additionally, mount the W&B artifacts cache as a persistent volume for specific use cases.