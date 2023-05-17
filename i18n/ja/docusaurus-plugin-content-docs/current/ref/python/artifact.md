# アーティファクト

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L110-L732)

データセットとモデルのバージョン管理に適した柔軟で軽量な構成要素。

```python
Artifact(
 name: str,
 type: str,
 description: Optional[str] = None,
 metadata: Optional[dict] = None,
 incremental: Optional[bool] = None,
 use_as: Optional[str] = None
) -> None
```

`add`系の関数を使って内容を追加できる空のアーティファクトを構築します。アーティファクトに必要なすべてのファイルが揃ったら、`wandb.log_artifact()`を呼び出してそのアーティファクトをログに記録できます。

| 引数 | |
| :--- | :--- |
| `name` | (str) このアーティファクトの人間に読みやすい名前。UIでアーティファクトを識別したり、`use_artifact`で参照するための名前です。名前には英字、数字、アンダースコア、ハイフン、ドットを使用できます。プロジェクト全体で一意である必要があります。 |
| `type` | (str) アーティファクトのタイプで、アーティファクトを整理・区別するために使用されます。一般的なタイプには「データセット」や「モデル」がありますが、英字、数字、アンダースコア、ハイフン、ドットを含む任意の文字列を使用できます。 |
| `description` | (str, optional) アーティファクトの説明を提供するフリーテキストです。説明はUIでマークダウン形式でレンダリングされるので、テーブルやリンクなどを記述するのに適しています。 |
| `metadata` | (dict, optional) アーティファクトに関連する構造化データ。たとえば、データセットのクラス分布があります。これは最終的にUIで検索可能で表示可能になります。合計キー数には100個の制限があります。 |

#### 例:

基本的な使い方
```
wandb.init()

```python
artifact = wandb.Artifact('mnist', type='dataset')
artifact.add_dir('mnist/')
wandb.log_artifact(artifact)
```

| Returns | |
| :--- | :--- |
| `Artifact`オブジェクト。 |





| Attributes | |
| :--- | :--- |
| `aliases` | このアーティファクトに関連付けられたエイリアス。リストは可変であり、`save()`を呼び出すと、エイリアスの変更がすべて永続化されます。 |
| `commit_hash` | このアーティファクトがコミットされたときに返されるハッシュ。 |
| `description` | アーティファクトの説明。 |
| `digest` | アーティファクトの論理的なダイジェスト。ダイジェストは、アーティファクトの内容のチェックサムです。アーティファクトが現在の`latest`バージョンと同じダイジェストを持っている場合、`log_artifact`は何も操作しません。 |
| `entity` | このアーティファクトが属しているエンティティの名前。 |
| `id` | アーティファクトのID。 |
| `manifest` | アーティファクトのマニフェスト。マニフェストは、アーティファクトのすべての内容をリストし、アーティファクトが記録された後は変更できません。 |
| `metadata` | ユーザー定義のアーティファクトメタデータ。 |
| `name` | アーティファクトの名前。 |
| `project` | このアーティファクトが属しているプロジェクトの名前。 |
| `size` | アーティファクトの合計サイズ（バイト単位）。 |
| `source_version` | このアーティファクトの親アーティファクトコレクションの下でのバージョンインデックス。"v{number}"という形式の文字列。 |
| `state` | アーティファクトの状態。"PENDING"、"COMMITTED"、または "DELETED" のいずれか。 |
| `type` | アーティファクトのタイプ。 |
| `version` | このアーティファクトのバージョン。たとえば、アーティファクトの最初のバージョンの場合、`version`は 'v0' になります。|



## メソッド

### `add`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L491-L572)

```python
add(
 obj: data_types.WBValue,
 name: str
) -> ArtifactManifestEntry
```

wandb.WBValue `obj`をアーティファクトに追加します。

```
obj = artifact.get(name)
```

| 引数 | |
| :--- | :--- |
| `obj` | (wandb.WBValue) 追加するオブジェクト。現在は、Bokeh、JoinedTable、PartitionedTable、Table、Classes、ImageMask、BoundingBoxes2D、Audio、Image、Video、Html、Object3D のいずれかをサポートしています。 |
| `name` | (str) オブジェクトを追加するアーティファクト内のパス。 |



| 戻り値 | |
| :--- | :--- |
| `ArtifactManifestEntry` | 追加されたマニフェストエントリ |



| 例外 | |
| :--- | :--- |
| `ArtifactFinalizedError` | アーティファクトが既に終了している場合。 |



#### 例：

基本的な使用方法
```
artifact = wandb.Artifact('my_table', 'dataset')
table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
artifact.add(table, "my_table")

wandb.log_artifact(artifact)
```

オブジェクトを取得する:
```
artifact = wandb.use_artifact('my_table:latest')
table = artifact.get("my_table")
```


### `add_dir`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L419-L452)

```python
add_dir(
 local_path: str,
 name: Optional[str] = None
) -> None
```

ローカルのディレクトリーをアーティファクトに追加します。


| 引数 | |
| :--- | :--- |
| `local_path` | (str) 追加されるディレクトリーへのパス。 |
| `name` | (str, 任意) 追加されるディレクトリーに使うアーティファクト内のパス。デフォルトでは、アーティファクトのルートになります。 |



#### 例:

明示的な名前のないディレクトリーを追加する:
```
# `my_dir/`内のすべてのファイルが、アーティファクトのルートに追加されます。
artifact.add_dir('my_dir/')
```
明示的にディレクトリを追加して名前を付ける:

```
# `my_dir/`内のすべてのファイルが`destination/`の下に追加される.
artifact.add_dir('my_dir/', name='destination')
```

| 例外 | |
| :--- | :--- |
| `ArtifactFinalizedError` | アーティファクトが既に確定している場合。|

| 戻り値 | |
| :--- | :--- |
| なし |

### `add_file`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L398-L417)

```python
add_file(
 local_path: str,
 name: Optional[str] = None,
 is_tmp: Optional[bool] = (False)
) -> ArtifactManifestEntry
```

ローカルファイルをアーティファクトに追加します。

| 引数 | |
| :--- | :--- |
| `local_path` | (str) 追加されるファイルへのパス。 |
| `name` | (str, 任意) 追加されるファイルに対してアーティファクト内で使用するパス。デフォルトはファイルのベース名。 |
| `is_tmp` | (bool, 任意) True の場合、ファイルは衝突を避けるために決まった方法でリネームされます。 (デフォルト: False) |

#### 例:

明示的な名前のないファイルを追加する:
```
# `file.txt` として追加する
artifact.add_file('path/to/file.txt')
```

明示的な名前を持つファイルを追加する:
```
# 'new/path/file.txt' として追加する
artifact.add_file('path/to/file.txt', name='new/path/file.txt')
```

| Raises | |
| :--- | :--- |
| `ArtifactFinalizedError` | アーティファクトがすでに確定されている場合。 |



| Returns | |
| :--- | :--- |
| `ArtifactManifestEntry` | 追加されたマニフェストエントリ |



### `add_reference`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L454-L489)

```python
add_reference(
 uri: Union[ArtifactManifestEntry, str],
 name: Optional[str] = None,
 checksum: bool = (True),
 max_objects: Optional[int] = None
) -> Sequence[ArtifactManifestEntry]
```
アーティファクトにURIで示す参照を追加します。

ファイルやディレクトリを追加するのとは異なり、参照はW&Bにアップロードされません。ただし、
アーティファクトが参照を含むかアップロードされたファイルを含むかに関係なく、
`download()` のようなアーティファクトメソッドが使用できます。

デフォルトでは、W&Bは以下のスキームに対して特別な処理を提供しています:

- http(s): サーバーによって返される `Content-Length` と `ETag` のレスポンスヘッダーによって、ファイルのサイズとダイジェストが推定されます。
- s3: チェックサムとサイズは、オブジェクトメタデータから取得されます。バケットのバージョン管理が有効になっている場合、バージョンIDも追跡されます。
- gs: チェックサムとサイズは、オブジェクトメタデータから取得されます。バケットのバージョン管理が有効になっている場合、バージョンIDも追跡されます。
- file: チェックサムとサイズは、ファイルシステムから取得されます。このスキームは、ファイルを追跡したいが必ずしもアップロードするわけではないNFS共有や外部にマウントされたボリュームに便利です。

その他のスキームでは、ダイジェストはURIのハッシュであり、サイズは空白のままです。

| 引数 | |
| :--- | :--- |
| `uri` | (str) 追加する参照のURIパス。Artifact.get_pathから返されるオブジェクトも含めることができ、他のアーティファクトのエントリへの参照を格納することができます。 |
| `name` | (str) この参照の内容をアーティファクト内に配置するパス checksum: (bool, optional) 参照URIにあるリソースのチェックサムを実行するかどうか。チェックサムを行うことで自動的な整合性検証が可能になりますが、アーティファクトの作成を高速化するために無効にすることもできます。(デフォルト: True) |
| `max_objects` | (int, optional) ディレクトリやバケットストアプレフィックスを指す参照を追加する際に考慮すべき最大オブジェクト数。S3やGCSでは、この制限はデフォルトで10,000ですが、他のURIスキームでは上限がありません。(デフォルト: None) |



| 例外 | |
| :--- | :--- |
| `ArtifactFinalizedError` | アーティファクトがすでに確定されている場合。 |



| 戻り値 | |
| :--- | :--- |
| List["ArtifactManifestEntry"]: 追加されたマニフェストエントリ。 |
#### 例:



#### HTTPリンクの追加:


```python
# `file.txt` をアーティファクトのルートに参照として追加します。
artifact.add_reference("http://myserver.com/file.txt")
```

明示的な名前を持たないS3プレフィックスの追加：
```python
# `prefix/`の下にあるすべてのオブジェクトがアーティファクトのルートに追加されます。
artifact.add_reference("s3://mybucket/prefix")
```

明示的な名前を持つGCSプレフィックスの追加：
```python
# `prefix/`の下にあるすべてのオブジェクトがアーティファクトのルートの`path/`の下に追加されます。
artifact.add_reference("gs://mybucket/prefix", name="path")
```

### `checkout`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L594-L598)

```python
checkout(
 root: Optional[str] = None
) -> str
```

指定したルートディレクトリをアーティファクトの内容で置き換えます。

警告: この操作は、アーティファクトに含まれていない`root`内のすべてのファイルを削除します。

| 引数 | |
| :--- | :--- |
| `root` | (str, optional) このアーティファクトのファイルで置換するディレクトリー。 |



| 戻り値 | |
| :--- | :--- |
| (str): チェックアウトされたコンテンツへのパス。 |



### `delete`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L647-L651)

```python
delete() -> None
```

このアーティファクトを削除し、関連するすべてのファイルをクリーンアップします。

注意: 削除は永久的であり、取り消すことはできません。

| 戻り値 | |
| :--- | :--- |
| None |



### `download`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L586-L592)

```python
download(
 root: Optional[str] = None,
 recursive: bool = (False)
) -> FilePathStr
```
アーティファクトの内容を指定されたルートディレクトリにダウンロードします。

注意: `root` にある既存のファイルはそのままになります。`root` の内容をアーティファクトと完全に一致させたい場合は、`download` を呼び出す前に `root` を明示的に削除してください。

| 引数 |  |
| :--- | :--- |
| `root` | (str, 任意) このアーティファクトのファイルをダウンロードするディレクトリ。 |
| `recursive` | (bool, 任意) True の場合、すべての依存アーティファクトが積極的にダウンロードされます。そうでない場合は、依存するアーティファクトが必要に応じてダウンロードされます。 |



| 戻り値 |  |
| :--- | :--- |
| (str) | ダウンロードされた内容へのパス。 |



### `finalize`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L688-L701)

```python
finalize() -> None
```

このアーティファクトを最終化し、それ以上の変更を禁止します。

`log_artifact` を呼び出すと自動的に行われます。

| 戻り値 |  |
| :--- | :--- |
| なし |
### `get`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L580-L584)

```python
get(
 name: str
) -> data_types.WBValue
```

アーティファクトの相対`name`で指定された場所にあるWBValueオブジェクトを取得します。

| 引数 | |
| :--- | :--- |
| `name` |（str）取得するアーティファクトの相対名 |


| 例外 | |
| :--- | :--- |
| `ArtifactNotLoggedError` | アーティファクトがログされていないか、runがオフラインの場合 |


#### 例:

基本的な使い方
```
# アーティファクトをログするRun
with wandb.init() as r:
 artifact = wandb.Artifact('my_dataset', type='dataset')
 table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
 artifact.add(table, "my_table")
 wandb.log_artifact(artifact)

# アーティファクトを使用するRun
with wandb.init() as r:
 artifact = r.use_artifact('my_dataset:latest')
 table = r.get('my_table')
```
### `get_added_local_path_name`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L664-L686)

```python
get_added_local_path_name(
 local_path: str
) -> Optional[str]
```

ローカルファイルシステムパスで追加されたファイルのアーティファクト相対名を取得します。

| 引数 | |
| :--- | :--- |
| `local_path` | (str) アーティファクト相対名に変換するローカルパス。 |


| 戻り値 | |
| :--- | :--- |
| `str` | アーティファクト相対名。 |



#### 例:

基本的な使い方
```
artifact = wandb.Artifact('my_dataset', type='dataset')
artifact.add_file('path/to/file.txt', name='artifact/path/file.txt')

# `artifact/path/file.txt`を返す:
name = artifact.get_added_local_path_name('path/to/file.txt')
```
### `get_path`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L574-L578)

```python
get_path(
 name: str
) -> ArtifactManifestEntry
```

アーティファクト相対の`name`にあるファイルへのパスを取得します。


| 引数 | |
| :--- | :--- |
| `name` | (str) 取得するアーティファクト相対名 |



| Raises（例外） | |
| :--- | :--- |
| `ArtifactNotLoggedError` | アーティファクトがログに記録されていない場合、またはrunがオフラインの場合 |



#### 例：

基本的な使い方
```
# アーティファクトをログに記録するRun
with wandb.init() as r:
 artifact = wandb.Artifact('my_dataset', type='dataset')
 artifact.add_file('path/to/file.txt')
 wandb.log_artifact(artifact)

# アーティファクトを使用するRun
with wandb.init() as r:
 artifact = r.use_artifact('my_dataset:latest')
 path = artifact.get_path('file.txt')

# これで 'file.txt'を直接ダウンロードできます:
path.download()
```

### `json_encode`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L703-L706)

```python
json_encode() -> Dict[str, Any]
```

### `link`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/interface/artifacts/artifact.py#L521-L533)

```python
link(
 target_path: str,
 aliases: Optional[List[str]] = None
) -> None
```

エイリアスを使用して、このアーティファクトをポートフォリオ（プロモートされたアーティファクトのコレクション）にリンクします。

| 引数 |  |
| :--- | :--- |
| `target_path` | (str) ポートフォリオへのパス。{portfolio}、{project}/{portfolio}、または{entity}/{project}/{portfolio}の形式でなければなりません。 |
| `aliases` | (Optional[List[str]]) 指定されたポートフォリオ内でアーティファクトを一意に識別する文字列のリスト。 |
| 返り値 | |
| :--- | :--- |
| なし |

### `logged_by`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L372-L376)

```python
logged_by() -> "wandb.apis.public.Run"
```

このアーティファクトを最初にログしたrunを取得します。

### `new_file`

[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L378-L396)

```python
@contextlib.contextmanager
new_file(
 name: str,
 mode: str = "w",
 encoding: Optional[str] = None
) -> Generator[IO, None, None]
```

新しい一時ファイルを開き、自動的にアーティファクトに追加されるようにします。

| 引数 | |
| :--- | :--- |
| `name` | (str) アーティファクトに追加される新しいファイルの名前。 |
| `mode` | (str, optional) 新しいファイルを開くモード。 |
| `encoding` | (str, optional) 新しいファイルを開く際のエンコーディング。
#### 例:

```
artifact = wandb.Artifact('my_data', type='dataset')
with artifact.new_file('hello.txt') as f:
 f.write('hello!')
wandb.log_artifact(artifact)
```



| 返り値 | |
| :--- | :--- |
| (file): 書き込み可能な新しいファイルオブジェクト。クローズすると、ファイルが自動的にアーティファクトに追加されます。 |



| 例外 | |
| :--- | :--- |
| `ArtifactFinalizedError` | アーティファクトがすでに確定している場合。 |



### `save`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L606-L645)

```python
save(
 project: Optional[str] = None,
 settings: Optional['wandb.wandb_sdk.wandb_settings.Settings'] = None
) -> None
```

アーティファクトに加えられた変更を永続化します。
現在runが実行中の場合、そのrunはこのアーティファクトをログに記録します。runが実行されていない場合は、このアーティファクトをトラッキングするために、 "auto" タイプのrunが作成されます。

| 引数 | |
| :--- | :--- |
| `project` | (str, 任意) runがまだコンテキスト設定に存在しない場合に、アーティファクトに使用するプロジェクト (wandb.Settings, 任意) 自動runの初期化時に使用する設定オブジェクト。主にテストハーネスで使用されます。 |



| 返り値 | |
| :--- | :--- |
| None |



### `used_by`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L366-L370)

```python
used_by() -> List['wandb.apis.public.Run']
```

このアーティファクトを使用したrunのリストを取得します。


### `verify`



[ソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L600-L604)

```python
verify(
 root: Optional[str] = None
) -> bool
```
アーティファクトの実際の内容がマニフェストと一致することを確認してください。

ディレクトリ内のすべてのファイルがチェックサムされ、それらのチェックサムはアーティファクトのマニフェストと照らし合わせられます。

注：参照は検証されません。

| 引数 | |
| :--- | :--- |
| `root` | (str, optional) 検証するディレクトリ。 Noneの場合、アーティファクトは './artifacts/self.name/' にダウンロードされます。 |



| Raises | |
| :--- | :--- |
| (ValueError): 検証に失敗した場合。|



### `wait`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_artifacts.py#L653-L662)

```python
wait(
 timeout: Optional[int] = None
) -> ArtifactInterface
```

アーティファクトのログが終了するのを待ちます。


| 引数 | |
| :--- | :--- |
| `timeout` | (int, optional) これまでに待ちます。 |
### `__getitem__`



[ソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/interface/artifacts/artifact.py#L553-L578)

```python
__getitem__(
 name: str
) -> Optional[WBValue]
```

アーティファクト相対の `name` にあるWBValueオブジェクトを取得します。


| 引数 | |
| :--- | :--- |
| `name` | (str) 取得するアーティファクトの相対名 |



| 例外 | |
| :--- | :--- |
| `ArtifactNotLoggedError` | アーティファクトがログに記録されていないか、runがオフラインの場合 |



#### 例:

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