---
title: Runs
menu:
  reference:
    identifier: ja-ref-python-public-api-runs
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L64-L273 >}}

project とオプションのフィルターに関連付けられた、反復可能な Runs のコレクション。

```python
Runs(
    client: "RetryingClient",
    entity: str,
    project: str,
    filters: Optional[Dict[str, Any]] = None,
    order: Optional[str] = None,
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

これは通常、`Api`.runs メソッドを介して間接的に使用されます。

| 属性 |  |
| :--- | :--- |

## メソッド

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L141-L173)

```python
convert_objects()
```

### `histories`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L175-L270)

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = "_step",
    format: Literal['default', 'pandas', 'polars'] = "default",
    stream: Literal['default', 'system'] = "default"
)
```

フィルター条件に適合するすべての run のサンプリングされた履歴 メトリクス を返します。

| arg |  |
| :--- | :--- |
| `samples` | (int, optional) run ごとに返すサンプル数 |
| `keys` | (list[str], optional) 特定の キー の メトリクス のみを返します |
| `x_axis` | (str, optional) この メトリクス を xAxis のデフォルトとして使用します。_step |
| `format` | (Literal, optional) データを返す形式。オプションは "default"、"pandas"、"polars" |
| `stream` | (Literal, optional) メトリクス の場合は "default"、マシン メトリクス の場合は "system" |

| 戻り値 |  |
| :--- | :--- |
| `pandas.DataFrame` | format="pandas" の場合、履歴 メトリクス の `pandas.DataFrame` を返します。 |
| `polars.DataFrame` | format="polars" の場合、履歴 メトリクス の `polars.DataFrame` を返します。dicts のリスト: format="default" の場合、run_id キー を含む履歴 メトリクス を含む dict のリストを返します。 |

### `next`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L52-L53)

```python
update_variables()
```

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| クラス変数 |  |
| :--- | :--- |
| `QUERY`<a id="QUERY"></a> |   |
