---
title: watch
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.19.5/wandb/sdk/wandb_run.py#L2808-L2843 >}}

Hooks into the given PyTorch model(s) to monitor gradients and the model's computational graph.

```python
watch(
    models: (torch.nn.Module | Sequence[torch.nn.Module]),
    criterion: (torch.F | None) = None,
    log: (Literal['gradients', 'parameters', 'all'] | None) = "gradients",
    log_freq: int = 1000,
    idx: (int | None) = None,
    log_graph: bool = (False)
) -> None
```

This function can track parameters, gradients, or both during training. It should be
extended to support arbitrary machine learning models in the future.

| Args |  |
| :--- | :--- |
|  `models` | A single model or a sequence of models to be monitored. | 
|`criterion `| The loss function being optimized (optional). |
|`log `| Specifies whether to log `gradients`, `parameters`, or `all`. Set to `None` to disable logging. (default="gradients") |
|`log_freq `|: How frequently to log gradients and parameters, expressed in batches. (default=1000)| 
|`idx `| Index used when tracking multiple models with `wandb.watch`. (default=None)| 
|`log_graph `| Whether to log the model's computational graph. (default=False) |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  If `wandb.init` has not been called or if any of the models are not instances of `torch.nn.Module`. |
