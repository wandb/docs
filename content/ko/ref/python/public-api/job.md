---
title: Job
menu:
  reference:
    identifier: ko-ref-python-public-api-job
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L30-L211 >}}

```python
Job(
    api: "Api",
    name,
    path: Optional[str] = None
) -> None
```

| 속성 |  |
| :--- | :--- |

## 메소드

### `call`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L167-L211)

```python
call(
    config, project=None, entity=None, queue=None, resource="local-container",
    resource_args=None, template_variables=None, project_queue=None, priority=None
)
```

### `set_entrypoint`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L164-L165)

```python
set_entrypoint(
    entrypoint: List[str]
)
```