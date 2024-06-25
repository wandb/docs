
# Run

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L272-L901' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースを見る</a></button></p>

エンティティおよびプロジェクトと関連付けられた単一のrun。

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

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L371-L411)

```python
@classmethod
create(
    api, run_id=None, project=None, entity=None
)
```

指定したプロジェクトのrunを作成します。

### `delete`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L529-L557)

```python
delete(
    delete_artifacts=(False)
)
```

wandbバックエンドから指定されたrunを削除します。

### `display`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/attrs.py#L15-L26)

```python
display(
    height=420, hidden=(False)
) -> bool
```

このオブジェクトをjupyterで表示します。

### `file`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L619-L629)

```python
file(
    name
)
```

アーティファクト内の指定された名前のファイルのパスを返します。

| 引数 |  |
| :--- | :--- |
|  name (str): リクエストされたファイルの名前。 |

| 戻り値 |  |
| :--- | :--- |
|  名前の引数に一致する `File`。 |

### `files`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L606-L617)

```python
files(
    names=None, per_page=50
)
```

指定された名前の各ファイルのファイルパスを返します。

| 引数 |  |
| :--- | :--- |
|  names (list): リクエストされたファイルの名前、空の場合はすべてのファイルを返します。 per_page (int): ページごとの結果数。 |

| 戻り値 |  |
| :--- | :--- |
|  `File`オブジェクトのイテレータである `Files`オブジェクト。 |

### `history`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L655-L695)

```python
history(
    samples=500, keys=None, x_axis="_step", pandas=(True), stream="default"
)
```

runのサンプル履歴メトリクスを返します。

履歴レコードがサンプリングされることを気にしないのであれば、これが簡単で高速です。

| 引数 |  |
| :--- | :--- |
|  `samples` |  (int, オプション) 返すサンプル数 |
|  `pandas` |  (bool, オプション) pandasのデータフレームを返すかどうか |
|  `keys` |  (list, オプション) 特定のキーのメトリクスのみを返す |
|  `x_axis` |  (str, オプション) これを xAxis として使用、デフォルトは _step |
|  `stream` |  (str, オプション) メトリクス用の "default"、マシンメトリクス用の "system" |

| 戻り値 |  |
| :--- | :--- |
|  `pandas.DataFrame` |  pandas=Trueの場合、履歴メトリクスの `pandas.DataFrame` を返します。 リスト辞書: pandas=Falseの場合、履歴メトリクスの辞書のリストを返します。 |

### `load`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L413-L477)

```python
load(
    force=(False)
)
```

### `log_artifact`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L794-L828)

```python
log_artifact(
    artifact, aliases=None
)
```

runの出力としてアーティファクトを宣言します。

| 引数 |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)`から返されたアーティファクト。 aliases (list, オプション): このアーティファクトに適用するエイリアス。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `logged_artifacts`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L753-L755)

```python
logged_artifacts(
    per_page=100
)
```

### `save`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L559-L560)

```python
save()
```

### `scan_history`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L697-L751)

```python
scan_history(
    keys=None, page_size=1000, min_step=None, max_step=None
)
```

runのすべての履歴レコードの反復可能なコレクションを返します。

#### 例:

サンプルrunのすべてのloss値をエクスポートする

```python
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```

| 引数 |  |
| :--- | :--- |
|  keys ([str], オプション): これらのキーのみをフェッチし、すべてのキーが定義されている行のみをフェッチします。 page_size (int, オプション): APIからフェッチするページのサイズ。 min_step (int, オプション): 一度にスキャンする最低ページ数。 max_step (int, オプション): 一度にスキャンする最大ページ数。 |

| 戻り値 |  |
| :--- | :--- |
|  履歴レコード（dict）を反復可能なコレクション。 |

### `snake_to_camel`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/attrs.py#L11-L13)

```python
snake_to_camel(
    string
)
```

### `to_html`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L887-L895)

```python
to_html(
    height=420, hidden=(False)
)
```

このrunを表示するiframeを含むHTMLを生成します。

### `update`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L502-L527)

```python
update()
```

変更をwandbバックエンドにrunオブジェクトとして保存します。

### `upload_file`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L631-L653)

```python
upload_file(
    path, root="."
)
```

ファイルをアップロードします。

| 引数 |  |
| :--- | :--- |
|  path (str): アップロードするファイルの名前。 root (str): ファイルを保存するルートパス。現在のディレクトリに "my_dir/file.txt" としてファイルを保存したい場合は、rootを "../" に設定します。 |

| 戻り値 |  |
| :--- | :--- |
|  名前の引数に一致する `File`。 |

### `use_artifact`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L761-L792)

```python
use_artifact(
    artifact, use_as=None
)
```

runの入力としてアーティファクトを宣言します。

| 引数 |  |
| :--- | :--- |
|  artifact (`Artifact`): `wandb.Api().artifact(name)`から返されたアーティファクト。 use_as (string, オプション): スクリプト内でアーティファクトの使用法を識別する文字列。ベータ版wandb launch機能のアーティファクト交換機能を使用する際に、runで使用されるアーティファクトを簡単に区別するために使用されます。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `used_artifacts`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L757-L759)

```python
used_artifacts(
    per_page=100
)
```

### `wait_until_finished`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/apis/public/runs.py#L479-L500)

```python
wait_until_finished()
```