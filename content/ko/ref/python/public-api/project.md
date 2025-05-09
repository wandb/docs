---
title: Project
menu:
  reference:
    identifier: ko-ref-python-public-api-project
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/projects.py#L79-L154 >}}

project는 runs를 위한 네임스페이스입니다.

```python
Project(
    client, entity, project, attrs
)
```

| 속성 |  |
| :--- | :--- |

## Methods

### `artifacts_types`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/projects.py#L112-L114)

```python
artifacts_types(
    per_page=50
)
```

### `display`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L16-L37)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Jupyter에서 이 오브젝트를 표시합니다.

### `snake_to_camel`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `sweeps`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/projects.py#L116-L154)

```python
sweeps()
```

### `to_html`

[소스 보기](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/projects.py#L96-L104)

```python
to_html(
    height=420, hidden=(False)
)
```

이 project를 표시하는 iframe을 포함하는 HTML을 생성합니다.
