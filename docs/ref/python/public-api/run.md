# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L272-L901' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

エンティティとプロジェクトに関連付けられた単一のRun。

```python
Run(
    client: "RetryingClient",
    entity: str,
    project: str,
    run_id: str,
    attrs: Optional[Mapping] = None,
    include_sweeps: bool = (True)
)
```

| 属性 |  |
| :--- | :--- |

## メソッド

### `create`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L371-L411)

```python
@classmethod
create(
    api, run_id=None, project=None, entity=None
)
```

指定されたプロジェクトのRunを作成します。

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L529-L557)

```python
delete(
    delete_artifacts=(False)
)
```

wandbのバックエンドから指定したRunを削除します。

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

このオブジェクトをjupyterで表示します。

### `file`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L619-L629)

```python
file(
    name
)
```

アーティファクト内の指定した名前のファイルのパスを返します。

| 引数 |  |
| :--- | :--- |
|  name (str): 要求されたファイルの名前。 |

| 戻り値 |  |
| :--- | :--- |
|  名前の引数に一致する `File`。 |

### `files`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L606-L617)

```python
files(
    names=None, per_page=50
)
```

指定した名前の各ファイルのファイルパスを返します。

| 引数 |  |
| :--- | :--- |
|  names (list): 要求されたファイルの名前。空の場合はすべてのファイルを返す。 per_page (int): ページあたりの結果の数。 |

| 戻り値 |  |
| :--- | :--- |
|  `File`オブジェクトのイテレータである`Files`オブジェクト。 |

### `history`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L655-L695)

```python
history(
    samples=500, keys=None, x_axis="_step", pandas=(True), stream="default"
)
```

Runのサンプリングされた履歴メトリクスを返します。

履歴レコードがサンプリングされていることに問題がなければ、これが簡単かつ迅速です。

| 引数 |  |
| :--- | :--- |
|  `samples` |  (int, オプション) 返すサンプルの数 |
|  `pandas` |  (bool, オプション) pandasデータフレームを返す |
|  `keys` |  (list, オプション) 特定のキーのメトリクスのみを返す |
|  `x_axis` |  (str, オプション) xAxisとして使用するメトリクス、デフォルトは_step |
|  `stream` |  (str, オプション) メトリクス用の"default"、マシンメトリクス用の"system" |

| 戻り値 |  |
| :--- | :--- |
|  `pandas.DataFrame` |  pandas=Trueの場合、履歴メトリクスの`pandas.DataFrame`を返します。 list of dicts: pandas=Falseの場合、履歴メトリクスのdictのリストを返します。 |

### `load`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L413-L477)

```python
load(
    force=(False)
)
```

### `log_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L794-L828)

```python
log_artifact(
    artifact, aliases=None
)
```

Runの出力としてアーティファクトを宣言します。

| 引数 |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)`から返されたArtifacts。 aliases (リスト, オプション): このアーティファクトに適用するエイリアスリスト。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact`オブジェクト。 |

### `logged_artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L753-L755)

```python
logged_artifacts(
    per_page=100
)
```

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L559-L560)

```python
save()
```

### `scan_history`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L697-L751)

```python
scan_history(
    keys=None, page_size=1000, min_step=None, max_step=None
)
```

Runのすべての履歴レコードの反復可能なコレクションを返します。

#### 例:

例のRunのすべての損失値をエクスポートします

```python
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```

| 引数 |  |
| :--- | :--- |
|  keys ([str], オプション): これらのキーのみを取得し、定義されているすべてのキーを持つ行のみを取得します。 page_size (int, オプション): APIから取得するページのサイズ。 min_step (int, オプション): 一度にスキャンするページの最小数。 max_step (int, オプション): 一度にスキャンするページの最大数。 |

| 戻り値 |  |
| :--- | :--- |
|  履歴レコード（dict）の反復可能なコレクション。 |

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L887-L895)

```python
to_html(
    height=420, hidden=(False)
)
```

このRunを表示するiframeを含むHTMLを生成します。

### `update`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L502-L527)

```python
update()
```

Runオブジェクトへの変更内容をwandbのバックエンドに記録します。

### `upload_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L631-L653)

```python
upload_file(
    path, root="."
)
```

ファイルをアップロードします。

| 引数 |  |
| :--- | :--- |
|  path (str): アップロードするファイルの名前。 root (str): ファイルを相対的に保存するルートパス。例：現在"my_dir"にいて、ファイルをRunに"my_dir/file.txt"として保存したい場合は、rootを"../"に設定します。 |

| 戻り値 |  |
| :--- | :--- |
|  名前の引数に一致する `File`。 |

### `use_artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L761-L792)

```python
use_artifact(
    artifact, use_as=None
)
```

Runの入力としてArtifactsを宣言します。

| 引数 |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)`から返されたArtifacts。 use_as (string, オプション): スクリプトでArtifactの使用方法を識別する文字列。beta wandbローンンチ機能のArtifact置換機能を使用する場合、Runで使用されるArtifactsを簡単に区別できます。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact`オブジェクト。 |

### `used_artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L757-L759)

```python
used_artifacts(
    per_page=100
)
```

### `wait_until_finished`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/apis/public/runs.py#L479-L500)

```python
wait_until_finished()
```