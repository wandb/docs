---
title: ジョブ
menu:
  reference:
    identifier: ja-ref-python-public-api-job
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L30-L211 >}}

```python
Job(
    api: "Api",
    name,
    path: Optional[str] = None
) -> None
```

| 属性 |  |
| :--- | :--- |

## メソッド

### `call`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L167-L211)

```python
call(
    config, project=None, entity=None, queue=None, resource="local-container",
    resource_args=None, template_variables=None, project_queue=None, priority=None
)
```

### `set_entrypoint`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/jobs.py#L164-L165)

```python
set_entrypoint(
    entrypoint: List[str]
)
```