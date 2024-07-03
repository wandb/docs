# Runs

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L61-L269' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

プロジェクトに関連付けられた run の反復可能なコレクションで、オプションのフィルターが適用されます。

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

これは一般的に `Api`.runs メソッドを介して間接的に使用されます。

| 属性 |  |
| :--- | :--- |

## メソッド

### `convert_objects`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L136-L168)

```python
convert_objects()
```

### `histories`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L170-L266)

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = "_step",
    format: Literal['default', 'pandas', 'polars'] = "default",
    stream: Literal['default', 'system'] = "default"
)
```

フィルター条件に一致するすべての run のサンプル歴史メトリクスを返します。

| 引数 |  |
| :--- | :--- |
|  `samples` |  (int, オプション) 1つの run ごとに返すサンプルの数 |
|  `keys` |  (list[str], オプション) 特定のキーのメトリクスのみを返す |
|  `x_axis` |  (str, オプション) x軸として使用するメトリクス、デフォルトは _step |
|  `format` |  (Literal, オプション) データを返す形式、オプションは "default"、"pandas"、"polars" |
|  `stream` |  (Literal, オプション) メトリクス用の "default"、マシンメトリクス用の "system" |

| 返却値 |  |
| :--- | :--- |
|  `pandas.DataFrame` |  format="pandas" の場合、歴史メトリクスの `pandas.DataFrame` を返す |
|  `polars.DataFrame` |  format="polars" の場合、歴史メトリクスの `polars.DataFrame` を返す |
|  辞書のリスト | format="default" の場合、run_id キーを含む歴史メトリクスの辞書のリストを返す |

### `next`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L52-L53)

```python
update_variables()
```

### `__getitem__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| クラス変数 |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |