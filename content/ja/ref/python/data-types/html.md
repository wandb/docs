---
title: Html
menu:
  reference:
    identifier: ja-ref-python-data-types-html
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/html.py#L18-L107 >}}

任意の HTML に対する Wandb クラス。

```python
Html(
    data: Union[str, 'TextIO'],
    inject: bool = (True)
) -> None
```

| Args |  |
| :--- | :--- |
|  `data` |  (文字列または io オブジェクト) wandb に表示する HTML。 |
|  `inject` |  (boolean) スタイルシートを HTML オブジェクトに追加します。False に設定すると、HTML は変更されずに渡されます。 |

## メソッド

### `inject_head`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/html.py#L59-L74)

```python
inject_head() -> None
```
```