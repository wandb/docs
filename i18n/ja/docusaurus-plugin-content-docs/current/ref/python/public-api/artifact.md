# アーティファクト

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4214-L5254)

wandbのアーティファクトです。

```python
Artifact(
 client, entity, project, name, attrs=None
)
```

ログされたアーティファクトで、すべての属性、それを使用するrunへのリンク、
それをログしたrunへのリンクが含まれています。

#### 例:

基本的な使い方
```
api = wandb.Api()
artifact = api.artifact('project/artifact:alias')

# アーティファクトに関する情報を取得する...
artifact.digest
artifact.aliases
```

アーティファクトを更新する
```
artifact = api.artifact('project/artifact:alias')

# 説明を更新する
artifact.description = 'My new description'
# メタデータキーを選択的に更新する
artifact.metadata["oldKey"] = "new value"

# メタデータを完全に置き換える
artifact.metadata = {"newKey": "new value"}

# エイリアスを追加する
artifact.aliases.append('best')

# エイリアスを削除する
artifact.aliases.remove('latest')

# エイリアスを完全に置き換える
artifact.aliases = ['replaced']

# すべてのアーティファクトの変更を保存する
artifact.save()
```

アーティファクトグラフの走査
```
artifact = api.artifact('project/artifact:alias')

# アーティファクトからグラフを上下に辿る:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()

# runからグラフを上下に辿る:
logged_artifacts = run.logged_artifacts()
used_artifacts = run.used_artifacts()
```

アーティファクトの削除
```
artifact = api.artifact('project/artifact:alias')
artifact.delete()
```
| 属性 | |
| :--- | :--- |
| `aliases` | このアーティファクトに関連付けられたエイリアス。 |
| `commit_hash` | このアーティファクトがコミットされたときに返されるハッシュ。 |
| `created_at` | アーティファクトが作成された時刻。 |
| `description` | アーティファクトの説明。 |
| `digest` | アーティファクトの論理ダイジェスト。ダイジェストはアーティファクトの内容のチェックサムです。アーティファクトが現在の`latest`バージョンと同じダイジェストを持っている場合、`log_artifact`は何もしません。 |
| `entity` | このアーティファクトが所属するエンティティの名前。 |
| `id` | アーティファクトのID。 |
| `manifest` | アーティファクトのマニフェスト。マニフェストにはその内容がすべて記載され、アーティファクトがログに記録されると変更できません。 |
| `metadata` | ユーザー定義のアーティファクトメタデータ。 |
| `name` | アーティファクトの名前。 |
| `project` | このアーティファクトが所属するプロジェクト名。 |
| `size` | アーティファクトの合計サイズ（バイト単位）。 |
| `source_version` | 親アーティファクトコレクションの下のアーティファクトのバージョンインデックス。"v{number}"という形式の文字列。 |
| `state` | アーティファクトの状態。以下のいずれか: "PENDING", "COMMITTED", "DELETED"。 |
| `type` | アーティファクトのタイプ。 |
| `updated_at` | アーティファクトが最後に更新された時刻。 |
| `version` | 指定されたアーティファクトコレクションの下のアーティファクトのバージョンインデックス。"v{number}"という形式の文字列。 |



## メソッド

### `add`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4649-L4650)

```python
add(
 obj, name
)
```

wandb.WBValue `obj` をアーティファクトに追加します。

```
obj = artifact.get(name)
```
| 引数 | |
| :--- | :--- |
| `obj` | (wandb.WBValue) 追加するオブジェクト。現在、Bokeh、JoinedTable、PartitionedTable、Table、Classes、ImageMask、BoundingBoxes2D、Audio、Image、Video、Html、Object3D のいずれかに対応しています。 |
| `name` | (str) アーティファクト内のオブジェクトを追加するパス。 |



| 戻り値 | |
| :--- | :--- |
| `ArtifactManifestEntry` | 追加されたマニフェストエントリ |



| 例外 | |
| :--- | :--- |
| `ArtifactFinalizedError` | アーティファクトがすでに確定している場合。 |



#### 例:

基本的な使い方
```
artifact = wandb.Artifact('my_table', 'dataset')
table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
artifact.add(table, "my_table")

wandb.log_artifact(artifact)
```

オブジェクトの取得:
```
artifact = wandb.use_artifact('my_table:latest')
table = artifact.get("my_table")
```


### `add_dir`
[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4643-L4644)

```python
add_dir(
 path, name=None
)
```

アーティファクトにローカルディレクトリを追加します。

| 引数 | 説明 |
| :--- | :--- |
| `local_path` | (str) 追加されるディレクトリのパス。 |
| `name` | (str, 任意) 追加されるディレクトリに使われるアーティファクト内のパス。デフォルトでは、アーティファクトのルートになります。 |



#### 例:

明示的な名前なしでディレクトリを追加:
```
# `my_dir/`内の全てのファイルがアーティファクトのルートに追加されます。
artifact.add_dir('my_dir/')
```

明示的な名前でディレクトリを追加:
```
# `my_dir/`内の全てのファイルが`destination/`の下に追加されます。
artifact.add_dir('my_dir/', name='destination')
```


| 例外 | 説明 |
| :--- | :--- |
| `ArtifactFinalizedError` | アーティファクトがすでに確定済みの場合。 |
| 戻り値 | |
| :--- | :--- |
| None |



### `add_file`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4640-L4641)

```python
add_file(
 local_path, name=None, is_tmp=(False)
)
```

ローカルファイルをアーティファクトに追加します。


| 引数 | |
| :--- | :--- |
| `local_path` | (str) 追加されるファイルへのパス。 |
| `name` | (str, 任意) 追加されるファイルに対してアーティファクト内で使用するパス。デフォルトでは、ファイルのベース名が使用されます。 |
| `is_tmp` | (bool, 任意) True の場合、ファイルは衝突を避けるために決定論的にリネームされます。 (デフォルト: False) |



#### 例:

明示的な名前がないファイルを追加する:
```
# `file.txt' として追加
artifact.add_file('path/to/file.txt')
```

明示的な名前を持つファイルを追加する:
```
# 'new/path/file.txt'として追加
artifact.add_file('path/to/file.txt', name='new/path/file.txt')
```

| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | アーティファクトがすでに確定されている場合。 |

| Returns | |
| :--- | :--- |
| `ArtifactManifestEntry` | 追加されたマニフェストエントリ |

### `add_reference`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4646-L4647)

```python
add_reference(
 uri, name=None, checksum=(True), max_objects=None
)
```

URIで示される参照をアーティファクトに追加します。

ファイルやディレクトリを追加するのとは異なり、参照はW&Bにアップロードされません。ただし、
アーティファクトに参照が含まれているかアップロードされたファイルが含まれているかに関わらず、
アーティファクトメソッド（`download()`など）が使用できます。

デフォルトで、W&Bは次のスキームに対して特別な処理を提供しています。

- http(s): サイズとダイジェストは、サーバーが返す`Content-Length`と`ETag`応答ヘッダーによって推定されます。
- s3: チェックサムとサイズは、オブジェクトのメタデータから取得されます。バケットの
 バージョン管理が有効になっている場合は、バージョンIDも追跡されます。
- gs: チェックサムとサイズは、オブジェクトのメタデータから取得されます。バケットの
 バージョン管理が有効になっている場合は、バージョンIDも追跡されます。
- file: チェックサムとサイズは、ファイルシステムから取得されます。このスキームは、
 追跡したいファイルが含まれているNFS共有や外部からマウントされたボリュームがある場合に
 役立ちますが、必ずしもアップロードする必要はありません。
他のスキームでは、ダイジェストはURIとサイズのハッシュで、サイズは空白のままです。

| 引数 |  |
| :--- | :--- |
| `uri` | (str) 追加する参照のURIパス。Artifact.get_pathから返されるオブジェクトを指定して、別のアーティファクトのエントリへの参照を保存できます。 |
| `name` | (str) この参照のコンテンツをアーティファクト内に配置するパス checksum: (bool, optional) 参照URIにあるリソースをチェックサムするかどうか。チェックサムは自動的な整合性検証が可能になるため、強く推奨されますが、アーティファクトの作成を高速化するために無効にすることもできます。(デフォルト: True) |
| `max_objects` | (int, optional) ディレクトリーやバケットストアのプレフィックスを指す参照を追加する際に考慮すべきオブジェクトの最大数。S3およびGCSでは、この上限はデフォルトで10,000ですが、他のURIスキームでは上限がありません。(デフォルト: None) |



| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | アーティファクトがすでに確定されている場合。 |



| Returns | |
| :--- | :--- |
| List["ArtifactManifestEntry"]: 追加されたマニフェストエントリ。 |



#### 例:



#### HTTPリンクの追加:


```python
# `file.txt`をアーティファクトのルートに参照として追加します。
artifact.add_reference("http://myserver.com/file.txt")
```

明示的な名前がないS3プレフィックスを追加:
```python
# `prefix/`以下のすべてのオブジェクトがアーティファクトのルートに追加されます。
artifact.add_reference("s3://mybucket/prefix")
```
明示的な名前でGCSプレフィックスを追加してください：
```python
# `prefix/`の下にある全てのオブジェクトがアーティファクトのルートの`path/`下に追加されます。
artifact.add_reference("gs://mybucket/prefix", name="path")
```

### `checkout`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4776-L4791)

```python
checkout(
 root=None
)
```

指定されたルートディレクトリーをアーティファクトの内容で置き換えます。

警告：アーティファクトに含まれていない`root`内のすべてのファイルが削除されます。

| 引数 | |
| :--- | :--- |
| `root` | (str, 任意) このアーティファクトのファイルで置き換えるディレクトリー。 |



| 戻り値 | |
| :--- | :--- |
| (str): チェックアウトした内容へのパス。 |



### `delete`

[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4596-L4635)
```python
delete(
 delete_aliases=(False)
)
```

アーティファクトとそのファイルを削除します。

#### 例:

Runがログしたすべての "model" アーティファクトを削除する:
```
runs = api.runs(path="my_entity/my_project")
for run in runs:
 for artifact in run.logged_artifacts():
 if artifact.type == "model":
 artifact.delete(delete_aliases=True)
```



| 引数 | |
| :--- | :--- |
| `delete_aliases` | (bool) True の場合、アーティファクトに関連するすべてのエイリアスを削除します。それ以外の場合、既存のエイリアスがあるアーティファクトでは例外が発生します。|
 


### `download`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4731-L4774)

```python
download(
 root=None, recursive=(False)
)
```
アーティファクトの内容を指定されたルートディレクトリにダウンロードします。

注意: `root` に既存のファイルはそのまま残ります。`root` の内容をアーティファクトと完全に一致させたい場合、`download` を呼び出す前に root を明示的に削除してください。

| 引数 | |
| :--- | :--- |
| `root` | (str, 任意) このアーティファクトのファイルをダウンロードするディレクトリ |
| `recursive` | (bool, 任意) True の場合、すべての依存アーティファクトが積極的にダウンロードされます。それ以外の場合、依存アーティファクトは必要に応じてダウンロードされます。 |



| 戻り値 | |
| :--- | :--- |
| (str): ダウンロードされた内容へのパス |



### `expected_type`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4502-L4542)

```python
@staticmethod
expected_type(
 client, name, entity_name, project_name
)
```

特定のアーティファクト名とプロジェクトの期待されるタイプを返します。

### `file`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4822-L4842)
以下のマークダウンテキストを翻訳してください。日本語に翻訳し、翻訳されたテキストだけを返してください。他には何も言わないでください。テキスト:

```python
file(
 root=None
)
```

指定された root によってディレクトリーに単一のファイルアーティファクトをダウンロードします。


| 引数 | |
| :--- | :--- |
| `root` | (str, 任意) ファイルを配置するルートディレクトリー。デフォルトは './artifacts/self.name/'。 |



| 戻り値 | |
| :--- | :--- |
| (str)：ダウンロードされたファイルのフルパス。 |



### `files`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L5089-L5100)

```python
files(
 names=None, per_page=50
)
```

このアーティファクトに保存されているすべてのファイルを反復処理します。


| 引数 | |
| :--- | :--- |
| `names` | (str のリスト, 任意) アーティファクトのルートに相対的なファイル名のパスをリスト表示する場合。 |
| `per_page` | (int, デフォルト50) 1回のリクエストで戻すファイルの数 |
| 戻り値 | |
| :--- | :--- |
| （`ArtifactFiles`）: `File`オブジェクトを含むイテレータ |



### `from_id`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4298-L4345)

```python
@classmethod
from_id(
 artifact_id: str,
 client: Client
)
```




### `get`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4703-L4729)

```python
get(
 name
)
```

アーティファクト相対`name`にあるWBValueオブジェクトを取得します。
以下のMarkdownテキストを翻訳してください。日本語に翻訳し、翻訳したテキストのみを返してください。他に何も言わずに。テキスト:

| 引数 | |
| :--- | :--- |
| `name` | (str) 取得するアーティファクトの相対名 |



| 発生するエラー | |
| :--- | :--- |
| `ArtifactNotLoggedError` | アーティファクトが記録されていないか、runがオフラインの場合 |



#### 例:

基本的な使い方
```
# アーティファクトをログするrun
with wandb.init() as r:
 artifact = wandb.Artifact('my_dataset', type='dataset')
 table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
 artifact.add(table, "my_table")
 wandb.log_artifact(artifact)

# アーティファクトを使用するrun
with wandb.init() as r:
 artifact = r.use_artifact('my_dataset:latest')
 table = r.get('my_table')
```


### `get_path`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4691-L4701)

```python
get_path(
 name
)
```
`name` に関連するアーティファクトのファイルへのパスを取得します。


| 引数 | |
| :--- | :--- |
| `name` | (str) 取得するアーティファクトの相対名 |



| 例外 | |
| :--- | :--- |
| `ArtifactNotLoggedError` | アーティファクトが記録されていないか、runがオフラインの場合 |



#### 例:

基本的な使い方
```
# アーティファクトを記録するRun
with wandb.init() as r:
 artifact = wandb.Artifact('my_dataset', type='dataset')
 artifact.add_file('path/to/file.txt')
 wandb.log_artifact(artifact)

# アーティファクトを使用するRun
with wandb.init() as r:
 artifact = r.use_artifact('my_dataset:latest')
 path = artifact.get_path('file.txt')

 # これで 'file.txt' を直接ダウンロードできます:
 path.download()
```


### `json_encode`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4864-L4865)
以下は、Markdownのテキストチャンクです。これを日本語に翻訳してください。他に何も言わずに、翻訳したテキストだけを返してください。テキスト：

```python
json_encode()
```




### `link`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4553-L4594)

```python
link(
 target_path, aliases=None
)
```

エイリアスを使って、このアーティファクトをポートフォリオ（プロモートされたアーティファクトのコレクション）にリンクします。


| 引数 | |
| :--- | :--- |
| `target_path` | (str) ポートフォリオへのパス。{portfolio}、{project}/{portfolio}または{entity}/{project}/{portfolio}の形式でなければなりません。 |
| `aliases` | (Optional[List[str]]) 指定されたポートフォリオ内でアーティファクトを一意に識別する文字列のリスト。 |



| 戻り値 | |
| :--- | :--- |
| なし |



### `logged_by`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L5218-L5254)
```python
logged_by()
```

このアーティファクトをログしたrunを取得します。


| 返り値 | |
| :--- | :--- |
| `Run` | このアーティファクトをログしたRunオブジェクト |



### `new_file`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4637-L4638)

```python
new_file(
 name, mode=None
)
```

新しい一時ファイルを開き、それを自動的にアーティファクトに追加します。


| 引数 | |
| :--- | :--- |
| `name` | (str) アーティファクトに追加される新しいファイルの名前。 |
| `mode` | (str, optional) 新しいファイルを開くモード。 |
| `encoding` | (str, optional) 新しいファイルを開くエンコーディング。 |



#### 例:

```
artifact = wandb.Artifact('my_data', type='dataset')
with artifact.new_file('hello.txt') as f:
 f.write('hello!')
wandb.log_artifact(artifact)
```
以下は、Markdownテキストのチャンクを翻訳してください。日本語に訳してください。それ以外のことは何も言わずに、翻訳したテキストのみを返してください。テキスト：

| 戻り値 | |
| :--- | :--- |
| (ファイル): 書き込み可能な新しいファイルオブジェクト。クローズ時に、ファイルは自動的にアーティファクトに追加されます。|



| 例外 | |
| :--- | :--- |
| `ArtifactFinalizedError` | もしアーティファクトがすでに確定されていた場合。|



### `save`



[ソースコードを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4867-L4930)

```python
save()
```

アーティファクトの変更をwandbバックエンドに永続化します。


### `used_by`



[ソースコードを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L5171-L5216)

```python
used_by()
```

このアーティファクトを直接使用しているrunsを取得します。
以下のマークダウンテキストを翻訳してください。日本語に翻訳し、翻訳されたテキストを返してください。他のことは言わないでください。テキスト：

| 戻り値 | |
| :--- | :--- |
| [Run]: このアーティファクトを使用するRunオブジェクトのリスト |



### `verify`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4793-L4820)

```python
verify(
 root=None
)
```

アーティファクトの実際の内容がマニフェストと一致することを確認します。

ディレクトリ内のすべてのファイルがチェックサムされ、そのチェックサムが
アーティファクトのマニフェストと照らし合わせられます。

注: 参照は確認されません。

| 引数 | |
| :--- | :--- |
| `root` | (str, オプション) 確認するディレクトリ。Noneの場合、アーティファクトは'./artifacts/self.name/'にダウンロードされます。 |



| Raises | |
| :--- | :--- |
| (ValueError)：検証が失敗した場合。 |



### `wait`
以下のMarkdownテキストを翻訳してください。日本語に翻訳し、そのテキストだけを返してください。それ以外のことは何も言わないでください。テキスト：

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L4932-L4933)

```python
wait()
```

必要に応じて、このアーティファクトのログ記録が終わるのを待ちます。


| 戻り値 | |
| :--- | :--- |
| アーティファクト |



### `__getitem__`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/interface/artifacts/artifact.py#L553-L578)

```python
__getitem__(
 name: str
) -> Optional[WBValue]
```

アーティファクト相対`name`の場所にあるWBValueオブジェクトを取得します。


| 引数 | |
| :--- | :--- |
| `name` | (str) 取得するアーティファクト相対名 |



| 例外 | |
| :--- | :--- |
| `ArtifactNotLoggedError` | アーティファクトがログに記録されていないか、runがオフラインの場合 |
#### 例：

基本的な使い方
```
artifact = wandb.Artifact('my_table', 'dataset')
table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
artifact["my_table"] = table

wandb.log_artifact(artifact)
```

オブジェクトの取得：
```
artifact = wandb.use_artifact('my_table:latest')
table = artifact["my_table"]
```



| クラス変数 | |
| :--- | :--- |
| `QUERY` | |