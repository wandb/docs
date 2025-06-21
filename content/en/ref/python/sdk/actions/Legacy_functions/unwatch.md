---
title: unwatch()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py >}}




### <kbd>function</kbd> `wandb.unwatch`

```python
wandb.unwatch(
    models: 'torch.nn.Module | Sequence[torch.nn.Module] | None' = None
) → None
```

Remove pytorch model topology, gradient and parameter hooks. 



**Args:**
 
 - `models`:  Optional list of pytorch models that have had watch called on them. 
