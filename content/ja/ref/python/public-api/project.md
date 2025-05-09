---
title: プロジェクト
menu:
  reference:
    identifier: ja-ref-python-public-api-project
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/projects.py#L79-L154 >}}

プロジェクトは、run のための名前空間です。

```python
Project(
    client, entity, project, attrs
)
```

| 属性 |  |
| :--- | :--- |

## メソッド

### `artifacts_types`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/projects.py#L112-L114)

```python
artifacts_types(
    per_page=50
)
```

### `display`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L16-L37)

```python
display(
    height=420, hidden=(False)
) -> bool
```

jupyter でこのオブジェクトを表示します。

### `snake_to_camel`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/attrs.py#L12-L14)

```python
snake_to_camel(
    string
)
```

### `sweeps`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/projects.py#L116-L154)

```python
sweeps()
```

### `to_html`

[ソースを見る](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/projects.py#L96-L104)

```python
to_html(
    height=420, hidden=(False)
)
```

このプロジェクトを表示する iframe を含む HTML を生成します。