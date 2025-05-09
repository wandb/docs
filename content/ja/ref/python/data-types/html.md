---
title: I'm sorry, it seems like there is no text to translate. Please provide the
  content you want to be translated.
menu:
  reference:
    identifier: ja-ref-python-data-types-html
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/html.py#L18-L107 >}}

任意の HTML 用の Wandb クラス。

```python
Html(
    data: Union[str, 'TextIO'],
    inject: bool = (True)
) -> None
```

| Args |  |
| :--- | :--- |
|  `data` |  (string または io オブジェクト) wandb に表示する HTML |
|  `inject` |  (boolean) HTML オブジェクトにスタイルシートを追加します。False に設定すると、HTML は変更されずに通過します。 |

## メソッド

### `inject_head`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/data_types/html.py#L59-L74)

```python
inject_head() -> None
```