
# Sweep

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/sweeps.py#L30-L240' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

sweepに関連付けられた一連のrun。

```python
Sweep(
    client, entity, project, sweep_id, attrs=None
)
```

### 例:

以下のようにインスタンス化します:

```
api = wandb.Api()
sweep = api.sweep(path/to/sweep)
```

| 属性 |  |
| :--- | :--- |
|  `runs` |  (`Runs`) runのリスト |
|  `id` |  (str) sweep id |
|  `project` |  (str) project名 |
|  `config` |  (str) sweep configurationの辞書 |
|  `state` |  (str) sweepの状態 |
|  `expected_run_count` |  (int) sweepの予想run数 |

## メソッド

### `best_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/sweeps.py#L125-L148)

```python
best_run(
    order=None
)
```

configで定義された指標または渡されたorderでソートされた最も良いrunを返します。

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

このオブジェクトをjupyterで表示します。

### `get`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/sweeps.py#L173-L222)

```python
@classmethod
get(
    client, entity=None, project=None, sid=None, order=None, query=None, **kwargs
)
```

クラウドバックエンドに対してクエリを実行します。

### `load`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/sweeps.py#L106-L114)

```python
load(
    force: bool = (False)
)
```

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/sweeps.py#L224-L232)

```python
to_html(
    height=420, hidden=(False)
)
```

このsweepを表示するiframeを含むHTMLを生成します。

| クラス変数 |  |
| :--- | :--- |
|  `LEGACY_QUERY`<a id="LEGACY_QUERY"></a> |   |
|  `QUERY`<a id="QUERY"></a> |   |