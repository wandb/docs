---
title: Job
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/2678738e59629208ad4770e3d36300a272147c05/wandb/apis/public/jobs.py#L30-L211 >}}

```python
Job(
    api: "Api",
    name,
    path: Optional[str] = None
) -> None
```

| Attributes |  |
| :--- | :--- |

## Methods

### `call`

[View source](https://www.github.com/wandb/wandb/tree/2678738e59629208ad4770e3d36300a272147c05/wandb/apis/public/jobs.py#L167-L211)

```python
call(
    config, project=None, entity=None, queue=None, resource="local-container",
    resource_args=None, template_variables=None, project_queue=None, priority=None
)
```

### `set_entrypoint`

[View source](https://www.github.com/wandb/wandb/tree/2678738e59629208ad4770e3d36300a272147c05/wandb/apis/public/jobs.py#L164-L165)

```python
set_entrypoint(
    entrypoint: List[str]
)
```
