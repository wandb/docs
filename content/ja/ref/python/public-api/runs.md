---
title: run
menu:
  reference:
    identifier: ja-ref-python-public-api-runs
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L64-L273 >}}

プロジェクトに関連付けられた runs の反復可能なコレクションとオプションフィルター。

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

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L141-L173)

```python
convert_objects()
```

### `histories`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/public/runs.py#L175-L270)

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = "_step",
    format: Literal['default', 'pandas', 'polars'] = "default",
    stream: Literal['default', 'system'] = "default"
)
```

フィルター条件に適合するすべての runs のサンプル履歴メトリクスを返します。

| 引数 |  |
| :--- | :--- |
|  `samples` |  (int, オプション) 各 run に対して返されるサンプルの数 |
|  `keys` |  (list[str], オプション) 特定のキーのメトリクスのみを返します |
|  `x_axis` |  (str, オプション) このメトリクスを x 軸として使用します。デフォルトは _step |
|  `format` |  (Literal, オプション) データを返すフォーマット、オプションは "default", "pandas", "polars" |
|  `stream` |  (Literal, オプション) メトリクスの "default", マシンメトリクスの "system" |

| 戻り値 |  |
| :--- | :--- |
|  `pandas.DataFrame` |  format="pandas" の場合、履歴メトリクスの `pandas.DataFrame` を返します。 |
|  `polars.DataFrame` |  format="polars" の場合、履歴メトリクスの `polars.DataFrame` を返します。リスト of dicts: format="default" の場合、履歴メトリクスを含む dicts のリストを run_id キー付きで返します。 |

### `next`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L52-L53)

```python
update_variables()
```

### `__getitem__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| クラス変数 |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |