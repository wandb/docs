---
title: Job
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/jobs.py#L36-L217" >}}

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/jobs.py#L173-L217)

```python
call(
    config, project=None, entity=None, queue=None, resource="local-container",
    resource_args=None, template_variables=None, project_queue=None, priority=None
)
```

### `set_entrypoint`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/apis/public/jobs.py#L170-L171)

```python
set_entrypoint(
    entrypoint: List[str]
)
```
