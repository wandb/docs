import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# unwatch

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py'/>




### <kbd>function</kbd> `wandb.unwatch`

```python
wandb.unwatch(
    models: 'torch.nn.Module | Sequence[torch.nn.Module] | None' = None
) → None
```

Remove pytorch model topology, gradient and parameter hooks. 



**Args:**
  models (torch.nn.Module | Sequence[torch.nn.Module]):  Optional list of pytorch models that have had watch called on them