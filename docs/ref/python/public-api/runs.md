
# Runs

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L61-L269' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

プロジェクトとオプションのフィルターに関連するrunの反復可能なコレクション。

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

これは通常、`Api`.runsメソッドを介して間接的に使用されます。

| 属性 |  |
| :--- | :--- |

## メソッド

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L136-L168)

```python
convert_objects()
```

### `histories`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L170-L266)

```python
histories(
    samples: int = 500,
    keys: Optional[List[str]] = None,
    x_axis: str = "_step",
    format: Literal['default', 'pandas', 'polars'] = "default",
    stream: Literal['default', 'system'] = "default"
)
```

フィルター条件に合うすべてのrunのサンプル履歴メトリクスを返します。

| 引数 |  |
| :--- | :--- |
|  `samples` |  (任意のint) runごとに返すサンプル数 |
|  `keys` |  (任意のlist[str]) 特定のキーのメトリクスのみ返します |
|  `x_axis` |  (任意のstr) x軸デフォルトとしてこのメトリクスを使用します。_step にデフォルト設定されています |
|  `format` |  (任意のLiteral) データを返す形式。選択肢は "default"、"pandas"、"polars" |
|  `stream` |  (任意のLiteral) メトリクス用の "default"、マシンメトリクス用の "system" |

| 戻り値 |  |
| :--- | :--- |
|  `pandas.DataFrame` |  format="pandas"の場合、`pandas.DataFrame`として履歴メトリクスを返します。 |
|  `polars.DataFrame` |  format="polars"の場合、`polars.DataFrame`として履歴メトリクスを返します。 |
|  list of dicts | format="default"の場合、run_idキー付きの履歴メトリクスを含む辞書リストを返します。 |

### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/paginator.py#L72-L79)

```python
next()
```

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/paginator.py#L52-L53)

```python
update_variables()
```

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/paginator.py#L65-L70)

```python
__getitem__(
    index
)
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/paginator.py#L26-L28)

```python
__iter__()
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/paginator.py#L30-L35)

```python
__len__()
```

| クラス変数 |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |