
# Artifact

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L90-L2349' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHubでソースを見る</a></button></p>

データセットおよびモデルのバージョン管理に柔軟かつ軽量なビルディングブロックです。

```python
Artifact(
    name: str,
    type: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    incremental: bool = (False),
    use_as: Optional[str] = None
) -> None
```

空の W&B Artifact を構築します。`add`で始まるメソッドを使ってartifactの内容を追加します。すべての必要なファイルをartifactに追加したら、`wandb.log_artifact()`を呼び出してログします。

| 引数 |  |
| :--- | :--- |
|  `name` |  アーティファクトの人間が読める名前。W&BアプリUIまたはプログラムで特定のアーティファクトを識別するために名前を使用します。`use_artifact`公開APIを使ってインタラクティブにアーティファクトを参照できます。名前には文字、数字、アンダースコア、ハイフン、ドットを含めることができます。名前はプロジェクト全体で一意でなければなりません。|
|  `type` |  アーティファクトのタイプ。アーティファクトを整理し区別するためにタイプを使用します。文字、数字、アンダースコア、ハイフン、ドットを含む任意の文字列を使用できます。一般的なタイプには`dataset`や`model`があります。アーティファクトをW&B Model Registryにリンクする場合は、タイプ文字列に`model`を含めます。|
|  `description` |  アーティファクトの説明。モデルまたはデータセットアーティファクトの場合、標準化されたチームモデルまたはデータセットカードのドキュメントを追加します。アーティファクトの説明は、`Artifact.description`属性またはW&BアプリUIでプログラム的に表示できます。W&Bはアプリ内で説明をmarkdown形式でレンダリングします。|
|  `metadata` |  アーティファクトに関する追加情報。メタデータをキーと値のペアの辞書として指定します。合計で100のキーを超えないようにする必要があります。|

| 戻り値 |  |
| :--- | :--- |
|  `Artifact`オブジェクト。 |

| 属性 |  |
| :--- | :--- |
|  `aliases` |  アーティファクトバージョンに割り当てられた1つ以上のセマンティックのフレンドリーな参照または識別用の「ニックネーム」のリスト。エイリアスはプログラム上で参照できる変更可能な参照です。W&BアプリUIまたはプログラム上でアーティファクトのエイリアスを変更できます。詳細については、[Create new artifact versions](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version)を参照してください。|
|  `collection` |  このアーティファクトが取得されたコレクション。コレクションはアーティファクトバージョンの順序付きグループです。このアーティファクトがポートフォリオ/リンクされたコレクションから取得された場合、そのコレクションが返され、アーティファクトバージョンが元になったコレクションではありません。アーティファクトが元となるコレクションはソースシーケンスとして知られています。|
|  `commit_hash` |  このアーティファクトがコミットされたときに返されるハッシュ。|
|  `created_at` |  アーティファクトが作成されたタイムスタンプ。|
|  `description` |  アーティファクトの説明。|
|  `digest` |  アーティファクトの論理ダイジェスト。ダイジェストはアーティファクトの内容のチェックサムです。アーティファクトのダイジェストが現在の`latest`バージョンと同じであれば、`log_artifact`は実行されません。|
|  `entity` |  セカンダリ（ポートフォリオ）アーティファクトコレクションのエンティティ名。|
|  `file_count` |  ファイルの数（参照を含む）。|
|  `id` |  アーティファクトのID。|
|  `manifest` |  アーティファクトのマニフェスト。マニフェストにはそのすべての内容がリストされており、アーティファクトがログに記録された後に変更することはできません。|
|  `metadata` |  ユーザー定義のアーティファクトメタデータ。アーティファクトに関連付けられた構造化データ。|
|  `name` |  セカンダリ（ポートフォリオ）コレクション内のアーティファクト名とバージョン。形式は{collection}:{alias}の文字列です。アーティファクトが保存される前は、バージョンは未定のため名前のみが含まれています。|
|  `project` |  セカンダリ（ポートフォリオ）アーティファクトコレクションのプロジェクト名。|
|  `qualified_name` |  セカンダリ（ポートフォリオ）コレクションのentity/project/nameです。|
|  `size` |  バイト単位のアーティファクトの総サイズ。このアーティファクトが追跡する任意の参照を含みます。|
|  `source_collection` |  アーティファクトのプライマリー（シーケンス）コレクション。|
|  `source_entity` |  プライマリー（シーケンス）アーティファクトコレクションのエンティティ名。|
|  `source_name` |  プライマリー（シーケンス）コレクション内のアーティファクト名とバージョン。形式は{collection}:{alias}の文字列です。アーティファクトが保存される前は、バージョンは未定のため名前のみが含まれています。|
|  `source_project` |  プライマリー（シーケンス）アーティファクトコレクションのプロジェクト名。|
|  `source_qualified_name` |  プライマリー（シーケンス）コレクションのentity/project/nameです。|
|  `source_version` |  プライマリー（シーケンス）コレクション内のアーティファクトのバージョン。形式は"v{number}"です。|
|  `state` |  アーティファクトのステータス。"PENDING"、"COMMITTED"、または"DELETED"のいずれか。|
|  `ttl` |  アーティファクトのタイム・ツー・リーブ（TTL）ポリシー。TTLポリシーの期間が経過するとまもなくアーティファクトが削除されます。`None`に設定されている場合、TTLポリシーは無効化され、チームデフォルトのTTLが存在する場合でも、アーティファクトは削除の対象とはなりません。アーティファクトは、チーム管理者がデフォルトのTTLを定義し、アーティファクトにカスタムポリシーが設定されていない場合、チームのデフォルトのTTLポリシーを継承します。|
|  `type` |  アーティファクトのタイプ。一般的なタイプは`dataset`または`model`です。|
|  `updated_at` |  アーティファクトが最後に更新された時間。|
|  `version` |  セカンダリ（ポートフォリオ）コレクション内のアーティファクトのバージョン。|

## メソッド

### `add`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1344-L1441)

```python
add(
    obj: data_types.WBValue,
    name: StrPath
) -> ArtifactManifestEntry
```

wandb.WBValue`オブジェクト`をアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `obj` |  追加するオブジェクト。現在サポートされているオブジェクトの種類は、Bokeh、JoinedTable、PartitionedTable、Table、Classes、ImageMask、BoundingBoxes2D、Audio、Image、Video、Html、Object3Dです。|
|  `name` |  アーティファクト内でオブジェクトを追加するパス。|

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンは最終化されているため、変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。|

### `add_dir`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1200-L1260)

```python
add_dir(
    local_path: str,
    name: Optional[str] = None,
    skip_cache: Optional[bool] = (False),
    policy: Optional[Literal['mutable', 'immutable']] = "mutable"
) -> None
```

ローカルディレクトリをアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `local_path` |  ローカルディレクトリのパス。|
|  `name` |  アーティファクト内のサブディレクトリ名。指定した名前はW&BアプリUIにアーティファクトの`type`にネストされて表示されます。デフォルトはアーティファクトのルートです。|
|  `skip_cache` |  `True`に設定すると、アップロード中にファイルがキャッシュにコピー/移動されません。|
|  `policy` |  "mutable" | "immutable"。デフォルトは"mutable"です。"mutable": アップロード中のファイルの破損を防ぐためにファイルの一時コピーを作成します。"immutable": 保護を無効にし、ユーザーがファイルを削除したり変更したりしないことに依存します。|

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンは最終化されているため、変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。|
|  `ValueError` |  ポリシーは"mutable"または"immutable"でなければなりません。|

### `add_file`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1154-L1198)

```python
add_file(
    local_path: str,
    name: Optional[str] = None,
    is_tmp: Optional[bool] = (False),
    skip_cache: Optional[bool] = (False),
    policy: Optional[Literal['mutable', 'immutable']] = "mutable"
) -> ArtifactManifestEntry
```

ローカルファイルをアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `local_path` |  追加されるファイルへのパス。|
|  `name` |  追加されるファイルのアーティファクト内のパス。デフォルトはファイルのベースネームです。|
|  `is_tmp` |  Trueの場合、ファイルは衝突を避けるために決定論的に名前が変更されます。|
|  `skip_cache` |  `True`に設定すると、アップロード後にファイルがキャッシュにコピーされません。|
|  `policy` |  "mutable" | "immutable"。デフォルトは"mutable"です。"mutable": アップロード中のファイルの破損を防ぐためにファイルの一時コピーを作成します。"immutable": 保護を無効にし、ユーザーがファイルを削除したり変更したりしないことに依存します。|

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンは最終化されているため、変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。|
|  `ValueError` |  ポリシーは"mutable"または"immutable"でなければなりません。|

### `add_reference`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1262-L1342)

```python
add_reference(
    uri: Union[ArtifactManifestEntry, str],
    name: Optional[StrPath] = None,
    checksum: bool = (True),
    max_objects: Optional[int] = None
) -> Sequence[ArtifactManifestEntry]
```

URIで示される参照をアーティファクトに追加します。

ファイルやディレクトリをアーティファクトに追加するのとは異なり、参照はW&Bにアップロードされません。詳細は、[Track external files](https://docs.wandb.ai/guides/artifacts/track-external-files)を参照してください。

デフォルトでは、以下のスキームがサポートされています：

- http(s): ファイルのサイズとダイジェストはサーバーから返される`Content-Length`および`ETag`レスポンスヘッダーで推測されます。
- s3: チェックサムとサイズはオブジェクトのメタデータから取得されます。バケットのバージョン管理が有効な場合、バージョンIDも追跡されます。
- gs: チェックサムとサイズはオブジェクトのメタデータから取得されます。バケットのバージョン管理が有効な場合、バージョンIDも追跡されます。
- https, `*.blob.core.windows.net`（Azure）と一致するドメイン: チェックサムとサイズはblobのメタデータから取得されます。ストレージアカウントのバージョン管理が有効な場合、バージョンIDも追跡されます。
- file: チェックサムとサイズはファイルシステムから取得されます。このスキームはNFS共有や他の外部マウントされたボリュームに含まれるファイルを追跡したいが、必ずしもアップロードするとは限らない場合に有用です。

他のスキームに対しては、ダイジェストはURIのハッシュに過ぎず、サイズは空のままです。

| 引数 |  |
| :--- | :--- |
|  `uri` |  追加する参照のURIパス。URIパスは、他のアーティファクトのエントリへの参照を保存するために`Artifact.get_entry`から返されるオブジェクトであることもできます。|
|  `name` |  この参照の内容を配置するアーティファクトのパス。|
|  `checksum` |  参照URIにあるリソースのチェックサムを実行するかどうか。チェックサムを実行することを強くお勧めします。これにより自動的な整合性検証が可能になります。チェックサムを実行しない場合、アーティファクトの作成が速くなりますが、参照ディレクトリは繰り返し処理されないため、ディレクトリ内のオブジェクトはアーティファクトに保存されません。チェックサムが無効な場合、参照オブジェクトを追加することを推奨します。|
|  `max_objects` |  ディレクトリまたはバケットストアのプレフィックスを指す参照を追加する場合に考慮する最大オブジェクト数。デフォルトでは、Amazon S3、GCS、Azure、ローカルファイルの最大オブジェクト数は10,000,000です。他

### `download`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1621-L1662)

```python
download(
    root: Optional[StrPath] = None,
    allow_missing_references: bool = (False),
    skip_cache: Optional[bool] = None,
    path_prefix: Optional[StrPath] = None
) -> FilePathStr
```

指定されたルートディレクトリーにアーティファクトの内容をダウンロードします。

`root` 内の既存ファイルは変更されません。`download` を呼び出す前に `root` を明示的に削除すると、`root` の内容がアーティファクトと完全に一致するようになります。

| 引数 |  |
| :--- | :--- |
|  `root` |  W&Bがアーティファクトのファイルを保存するディレクトリー。 |
|  `allow_missing_references` |  `True` に設定すると、参照ファイルのダウンロード時に無効な参照パスが無視されます。 |
|  `skip_cache` |  `True` に設定すると、ダウンロード時にアーティファクトのキャッシュをスキップし、W&Bは各ファイルをデフォルトのルートまたは指定されたダウンロードディレクトリーにダウンロードします。 |

| 戻り値 |  |
| :--- | :--- |
|  ダウンロードされた内容のパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合に発生します。 |

### `file`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1950-L1975)

```python
file(
    root: Optional[str] = None
) -> StrPath
```

単一ファイルのアーティファクトを、`root` で指定したディレクトリーにダウンロードします。

| 引数 |  |
| :--- | :--- |
|  `root` |  ファイルを保存するルートディレクトリー。デフォルトは './artifacts/self.name/' です。 |

| 戻り値 |  |
| :--- | :--- |
|  ダウンロードされたファイルのフルパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合に発生します。 |
|  `ValueError` |  アーティファクトに複数のファイルが含まれている場合に発生します。 |

### `files`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1977-L1994)

```python
files(
    names: Optional[List[str]] = None,
    per_page: int = 50
) -> ArtifactFiles
```

このアーティファクトに保存されているすべてのファイルを繰り返し処理します。

| 引数 |  |
| :--- | :--- |
|  `names` |  一覧表示したいアーティファクトのルートに対するファイル名のパス。 |
|  `per_page` |  リクエストごとに返されるファイルの数。 |

| 戻り値 |  |
| :--- | :--- |
|  `File` オブジェクトを含むイテレーター。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合に発生します。 |

### `finalize`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L723-L731)

```python
finalize() -> None
```

アーティファクトバージョンを確定します。

アーティファクトが特定のバージョンとしてログされるため、一旦確定されたアーティファクトバージョンは変更できません。新しいアーティファクトバージョンを作成してさらにデータをログしてください。アーティファクトは `log_artifact` でログされると自動的に確定されます。

### `get`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1537-L1583)

```python
get(
    name: str
) -> Optional[data_types.WBValue]
```

アーティファクト相対 `name` に位置する WBValue オブジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクト相対名。 |

| 戻り値 |  |
| :--- | :--- |
|  `wandb.log()` でログされ、W&B UIで視覚化できる W&B オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていないか、run がオフラインの場合に発生します。 |

### `get_added_local_path_name`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1585-L1597)

```python
get_added_local_path_name(
    local_path: str
) -> Optional[str]
```

ローカルファイルシステムパスによって追加されたファイルのアーティファクト相対名を取得します。

| 引数 |  |
| :--- | :--- |
|  `local_path` |  アーティファクト相対名に解決するローカルパス。 |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクト相対名。 |

### `get_entry`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1515-L1535)

```python
get_entry(
    name: StrPath
) -> ArtifactManifestEntry
```

指定された名前のエントリーを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクト相対名。 |

| 戻り値 |  |
| :--- | :--- |
|  `W&B` オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていないか、run がオフラインの場合に発生します。 |
|  `KeyError` |  指定された名前のエントリーがアーティファクトに含まれていない場合に発生します。 |

### `get_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1507-L1513)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

非推奨。`get_entry(name)` を使用してください。

### `is_draft`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L741-L746)

```python
is_draft() -> bool
```

アーティファクトが保存されていないかどうかを確認します。

戻り値: ブール値。アーティファクトが保存されていない場合は `True`、保存された場合は `False`。

### `json_encode`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L2222-L2229)

```python
json_encode() -> Dict[str, Any]
```

アーティファクトをJSON形式にエンコードしたものを返します。

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトの属性を表す `string` キーを持つ `dict`。 |

### `link`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L2058-L2086)

```python
link(
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

このアーティファクトをポートフォリオ（プロモートされたアーティファクトコレクション）にリンクします。

| 引数 |  |
| :--- | :--- |
|  `target_path` |  プロジェクト内のポートフォリオへのパス。ターゲットパスは次のいずれかのスキーマに従う必要があります: `{portfolio}`, `{project}/{portfolio}` または `{entity}/{project}/{portfolio}`。アーティファクトをプロジェクト内の一般的なポートフォリオではなくモデルレジストリにリンクする場合は、ターゲットパスを次のスキーマに設定します: `{"model-registry"}/{Registered Model Name}` または `{entity}/{"model-registry"}/{Registered Model Name}`。 |
|  `aliases` |  指定されたポートフォリオ内でアーティファクトを一意に識別する文字列のリスト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合に発生します。 |

### `logged_by`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L2177-L2220)

```python
logged_by() -> Optional[Run]
```

アーティファクトを最初にログした W&B の run を取得します。

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトを最初にログした W&B の run の名前。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合に発生します。 |

### `new_draft`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L345-L377)

```python
new_draft() -> "Artifact"
```

このコミットされたアーティファクトと同じ内容で新しいドラフトアーティファクトを作成します。

返されたアーティファクトは拡張または変更され、新しいバージョンとしてログされることができます。

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合に発生します。 |

### `new_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1115-L1152)

```python
@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "w",
    encoding: Optional[str] = None
) -> Generator[IO, None, None]
```

新しい一時ファイルを開き、それをアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  アーティファクトに追加する新しいファイルの名前。 |
|  `mode` |  新しいファイルを開くためのファイルアクセスモード。 |
|  `encoding` |  新しいファイルを開くときに使用されるエンコーディング。 |

| 戻り値 |  |
| :--- | :--- |
|  書き込み可能な新しいファイルオブジェクト。閉じると自動的にアーティファクトに追加されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンが確定されたため、変更できません。代わりに新しいアーティファクトバージョンをログしてください。 |

### `path_contains_dir_prefix`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1664-L1671)

```python
@classmethod
path_contains_dir_prefix(
    path: StrPath,
    dir_path: StrPath
) -> bool
```

`path` が `dir_path` をプレフィックスとして含んでいる場合に `true` を返します。

### `remove`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1476-L1505)

```python
remove(
    item: Union[StrPath, 'ArtifactManifestEntry']
) -> None
```

アイテムをアーティファクトから削除します。

| 引数 |  |
| :--- | :--- |
|  `item` |  削除するアイテム。特定のマニフェストエントリーまたはアーティファクト相対パスの名前を指定できます。アイテムがディレクトリーに一致する場合、そのディレクトリー内のすべてのアイテムが削除されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンが確定されたため、変更できません。代わりに新しいアーティファクトバージョンをログしてください。 |
|  `FileNotFoundError` |  アイテムがアーティファクト内に存在しない場合に発生します。 |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L751-L790)

```python
save(
    project: Optional[str] = None,
    settings: Optional['wandb.wandb_sdk.wandb_settings.Settings'] = None
) -> None
```

アーティファクトに対して行った変更を保存します。

現在 run 中の場合、その run がこのアーティファクトをログします。現在 run 中でない場合は、アーティファクトを追跡するために "auto" タイプの run が作成されます。

| 引数 |  |
| :--- | :--- |
|  `project` |  run が既に存在しない場合にアーティファクトに使用するプロジェクト。 |
|  `settings` |  自動 run を初期化する際に使用する設定オブジェクト。主にテストハーネスで使用されます。 |

### `should_download_entry`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1673-L1679)

```python
@classmethod
should_download_entry(
    entry: ArtifactManifestEntry,
    prefix: Optional[StrPath]
) -> bool
```

### `unlink`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L2088-L2104)

```python
unlink() -> None
```

このアーティファクトが現在ポートフォリオ（プロモートされたアーティファクトコレクション）のメンバーである場合、リンクを解除します。

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合に発生します。 |
|  `ValueError` |  アーティファクトがリンクされていない場合、つまりポートフォリオコレクションのメンバーでない場合に発生します。 |

### `used_by`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L2130-L2175)

```python
used_by() -> List[Run]
```

このアーティファクトを使用した run のリストを取得します。

| 戻り値 |  |
| :--- | :--- |
|  `Run` オブジェクトのリスト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合に発生します。 |

### `verify`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1909-L1948)

```python
verify(
    root: Optional[str] = None
) -> None
```

アーティファクトの内容がマニフェストと一致するかどうかを検証します。

ディレクトリー内のすべてのファイルのチェックサムを計算し、それをアーティファクトのマニフェストとクロスチェックします。参照は検証されません。

| 引数 |  |
| :--- | :--- |
|  `root` |  検証するディレクトリー。None の場合、アーティファクトは './artifacts/self.name/' にダウンロードされます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合に発生します。 |
|  `ValueError` |  検証に失敗した場合に発生します。 |

### `wait

### `__setitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1099-L1113)

```python
__setitem__(
    name: str,
    item: data_types.WBValue
) -> ArtifactManifestEntry
```

`name` のパスに `item` をアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  オブジェクトを追加するアーティファクト内のパス。 |
|  `item` |  追加するオブジェクト。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンは確定されているため、変更を加えることはできません。代わりに、新しいアーティファクトバージョンをログに記録してください。 |