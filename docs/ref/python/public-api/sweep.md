# Sweep

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/sweeps.py#L30-L240' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

ある sweep に関連付けられた一連の runs。

```python
Sweep(
    client, entity, project, sweep_id, attrs=None
)
```

#### 例:

以下を使用してインスタンス化します:

```
api = wandb.Api()
sweep = api.sweep(path/to/sweep)
```

| 属性 |  |
| :--- | :--- |
|  `runs` |  (`Runs`) 複数の runs のリスト |
|  `id` |  (str) sweep id |
|  `project` |  (str) project の名前 |
|  `config` |  (str) sweep configuration の辞書 |
|  `state` |  (str) sweep の状態 |
|  `expected_run_count` |  (int) sweep の予想される runs の数 |

## メソッド

### `best_run`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/sweeps.py#L125-L148)

```python
best_run(
    order=None
)
```

設定で定義されたメトリックまたは渡された order によってソートされた最良の run を返します。

### `display`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

このオブジェクトを jupyter に表示します。

### `get`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/sweeps.py#L173-L222)

```python
@classmethod
get(
    client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

クラウドバックエンドに対してクエリを実行します。

### `load`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/sweeps.py#L106-L114)

```python
load(
    force: bool = (False)
)
```

### `snake_to_camel`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/sweeps.py#L224-L232)

```python
to_html(
    height=420, hidden=(False)
)
```

この sweep を表示する iframe を含む HTML を生成します。

| クラス変数 |  |
| :--- | :--- |
|  `LEGACY_QUERY`<a id="LEGACY_QUERY"></a> |   |
|  `QUERY`<a id="QUERY"></a> |   |