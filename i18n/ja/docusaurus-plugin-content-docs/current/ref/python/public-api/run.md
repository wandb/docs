# Run

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1664-L2276)

エンティティとプロジェクトに関連する単一のrunです。

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
| 属性 | |
| :--- | :--- |

## メソッド

### `create`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1761-L1801)

```python
@classmethod
create(
 api, run_id=None, project=None, entity=None
)
```

指定されたプロジェクトのrunを作成します。
### `delete`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1914-L1946)

```python
delete(
 アーティファクトを削除する=(False)
)
```

wandbバックエンドから指定されたrunを削除します。
### `display`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L979-L990)

```python
display(
 height=420, hidden=(False)
) -> bool
```

このオブジェクトをjupyterで表示します。
### `file`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2009-L2019)

```python
file(
 name
)
```

アーティファクト内の指定された名前のファイルのパスを返します。

| 引数 | |
| :--- | :--- |
| name (str): 要求されたファイルの名前。|

| 戻り値 | |
| :--- | :--- |
| 引数の名前に一致する`File`。 |



### `files`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1996-L2007)

```python
files(
 names=None, per_page=50
)
```
各ファイルの名前を返します。


| 引数 | |
| :--- | :--- |
| names (list): 要求されたファイルの名前、空の場合はすべてのファイルを返します per_page (int): 1ページあたりの結果数。|



| 戻り値 | |
| :--- | :--- |
| `File` オブジェクトを順に返す `Files` オブジェクト。|
### `history`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2045-L2085)

```python
history(
 サンプル=500, キー=None, x_axis="_step", pandas=(True), ストリーム="デフォルト"
)
```

Runのサンプル履歴メトリクスを返します。

履歴レコードがサンプリングされている場合、これはよりシンプルで高速です。
| 引数 | |
| :--- | :--- |
| `samples` | (int, 任意) 返すサンプル数 |
| `pandas` | (bool, 任意) pandasのデータフレームを返す |
| `keys` | (list, 任意) 特定のキーのメトリクスのみを返す |
| `x_axis` | (str, 任意) このメトリクスをx軸に使う（デフォルトは_step） |
| `stream` | (str, 任意) メトリクスの場合は"default"、機械メトリクスの場合は"system" |



| 戻り値 | |
| :--- | :--- |
| `pandas.DataFrame` | pandas=Trueの場合、履歴メトリクスの`pandas.DataFrame`を返す。リストの辞書：pandas=Falseの場合、履歴メトリクスの辞書のリストを返す。

### `load`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1803-L1861)

```python
load(
 force=(False)
)
```
### `log_artifact`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2183-L2215)

```python
log_artifact(
 artifact, aliases=None
)
```

アーティファクトをrunの出力として宣言します。
| 引数 | |
| :--- | :--- |
| artifact（`Artifact`）: `wandb.Api().artifact(name)`から返されるアーティファクト　エイリアス（リスト、オプショナル）: このアーティファクトに適用するエイリアス |



| 戻り値 | |
| :--- | :--- |
| `Artifact`オブジェクト。 |



### `logged_artifacts`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2142-L2144)

```python
logged_artifacts(
 per_page=100
)
```




### `save`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1948-L1949)

```python
save()
```

### `scan_history`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2087-L2140)

```python
scan_history(
 keys=None, page_size=1000, min_step=None, max_step=None
)
```

runのすべての履歴レコードを含む反復可能なコレクションを返します。


#### 例:

例のrunにおいて全ての損失値をエクスポートする

```python
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```




| 引数 | |
| :--- | :--- |
| keys ([str], 任意): これらのキーのみをフェッチし、また、すべてのキーが定義されている行のみをフェッチします。page_size (int, 任意): APIからフェッチするページのサイズ |

| 戻り値 | |
| :--- | :--- |
| 履歴レコード（dict）をイテレートするコレクション。 |



### `snake_to_camel`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L975-L977)

```python
snake_to_camel(
 文字列
)
```
### `to_html`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2262-L2270)

```python
to_html(
 高さ=420, 非表示=(False)
)
```
このrunを表示するiframeを含むHTMLを生成します。

### `update`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1886-L1912)

```python
update()
```

Runオブジェクトの変更をwandbのバックエンドに保存します。
### `upload_file`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2021-L2043)

```python
upload_file(
 パス, root="."
)
```

ファイルをアップロードします。
| 引数 | |
| :--- | :--- |
| path（文字列）: アップロードするファイル名。root（文字列）: ファイルを保存する相対パスのルート。例えば、"my_dir/file.txt" にファイルを保存したい場合で、現在 "my_dir" にいる場合は、rootを "../" に設定します。 |



| 戻り値 | |
| :--- | :--- |
| name 引数に一致する `File`。|



### `use_artifact`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2150-L2181)

```python
use_artifact(
 artifact, use_as=None
)
```

アーティファクトをrunの入力として宣言します。

| 引数 | |
| :--- | :--- |
| artifact (`Artifact`): `wandb.Api().artifact(name)`から返されるアーティファクト use_as (文字列, オプション): スクリプトでアーティファクトの使い方を識別する文字列。betaのwandb launch機能のアーティファクト交換機能を使用して、runで使用されるアーティファクトを簡単に区別できます。 |

| 返り値 | |
| :--- | :--- |
| `Artifact`オブジェクト。 |



### `used_artifacts`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2146-L2148)

```python
used_artifacts(
 per_page=100
)
```
### `wait_until_finished`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1863-L1884)

```python
wait_until_finished()
```