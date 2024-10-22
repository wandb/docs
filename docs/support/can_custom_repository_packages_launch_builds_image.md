---
title: "Can you use a custom repository for packages when Launch builds the image?"
tags:
   - launch
---

Yes. To do so, add the following line to your `requirements.txt` and replace the values passed to `index-url` and `extra-index-url` with your own values:

```text
----index-url=https://xyz@<your-repo-host> --extra-index-url=https://pypi.org/simple
```
 
The `requirements.txt` needs to be defined at the base root of the job.

## Automatic run re-queuing on preemption

In some cases, it can be useful to set up jobs to be resumed after they are interrupted.  For example, you might run broad hyperparameter sweeps on spot instances, and want them to pick up again when more spot instances spin up.  Launch can support this configuration on Kubernetes clusters.

If your Kubernetes queue is running a job on a node that’s pre-empted by a scheduler, the job will be automatically added back to the end of the queue so it can resume later. This resumed run will have the same name as the original, and can be followed from the same page in the UI as the original. A job can be automatically re-queued this way up to five times. 

Launch detects whether a pod is preempted by a scheduler by checking if the pod has the condition `DisruptionTarget` with one of the following reasons:

- `EvictionByEvictionAPI`
- `PreemptionByScheduler`
- `TerminationByKubelet`

If your job’s code is structured to allow resuming, it will enable these re-queued runs to pick up where they left off. Otherwise, runs will start from the beginning when they are re-queued. See our guide for [resuming runs](../guides/runs/resuming.md) for more info.   

There is currently no way to opt out of automatic run re-queuing for preempted nodes. However, if you delete a run from the UI or delete the node directly, it will not be re-queued.

Automatic run re-queuing is currently only available on Kubernetes queues; Sagemaker and Vertex are not yet supported.