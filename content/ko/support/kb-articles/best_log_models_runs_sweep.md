---
menu:
  support:
    identifier: ko-support-kb-articles-best_log_models_runs_sweep
support:
- artifacts
- sweeps
title: How do I best log models from runs in a sweep?
toc_hide: true
type: docs
url: /support/:filename
---

One effective approach for logging models in a [sweep]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) involves creating a model artifact for the sweep. Each version represents a different run from the sweep. Implement it as follows:

```python
wandb.Artifact(name="sweep_name", type="model")
```