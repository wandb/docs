# Artifact

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L90-L2356' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

データセットとモデルのバージョン管理のための柔軟で軽量なビルディングブロック。

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

空のW&B Artifactを構築します。`add`で始まるメソッドを使ってアーティファクトの内容を追加します。すべての必要なファイルがアーティファクトに追加されたら、`wandb.log_artifact()`を呼び出してログを作成できます。

| 引数 |  |
| :--- | :--- |
|  `name` |  アーティファクトの人間が読める名前。W&BアプリのUIまたはプログラムを使用して特定のアーティファクトを識別するために使用します。Public APIの`use_artifact`を使ってアーティファクトを参照できます。名前には文字、数字、アンダースコア、ハイフン、およびドットを含めることができます。名前はプロジェクト全体で一意である必要があります。 |
|  `type` |  アーティファクトのタイプ。アーティファクトのタイプを使用してアーティファクトを整理し、区別します。文字、数字、アンダースコア、ハイフン、およびドットを含む任意の文字列を使用できます。一般的なタイプには`dataset`や`model`が含まれます。アーティファクトをW&Bモデルレジストリにリンクしたい場合は、タイプ文字列に`model`を含めてください。 |
|  `description` |  アーティファクトの説明。モデルまたはデータセットのArtifactsの場合、標準化されたチームモデルまたはデータセットカードのドキュメントを追加します。アーティファクトの説明をプログラムによっては`Artifact.description`属性で、またはW&BアプリのUIからプログラム的に確認できます。W&Bはアプリ内で説明をMarkdownとしてレンダリングします。 |
|  `metadata` |  アーティファクトに関する追加情報。メタデータをキーと値のペアで辞書として指定します。キーの総数は100個以内で指定してください。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact`オブジェクト。 |

| 属性 |  |
| :--- | :--- |
|  `aliases` |  アーティファクトバージョンに割り当てられた、意味的に親しみやすい参照や識別用の「ニックネーム」のリスト。エイリアスはプログラムによって参照できる可変参照です。エイリアスはW&BアプリのUIまたはプログラムによって変更可能です。詳細については、[新しいアーティファクトバージョンの作成](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version)を参照してください。 |
|  `collection` |  このアーティファクトが取得されたコレクション。コレクションはアーティファクトバージョンの順序付けられたグループです。 |
|  `commit_hash` |  このアーティファクトがコミットされた時に返されるハッシュ。 |
|  `created_at` |  アーティファクトが作成されたタイムスタンプ。 |
|  `description` |  アーティファクトの説明。 |
|  `digest` |  アーティファクトの論理的ダイジェスト。 |
|  `entity` |  二次（ポートフォリオ）アーティファクトコレクションのエンティティの名前。 |
|  `file_count` |  ファイルの数（参照を含む）。 |
|  `id` |  アーティファクトのID。 |
|  `manifest` |  アーティファクトのマニフェスト。マニフェストはアーティファクトのすべての内容を一覧表示し、一度ログに記録された後は変更できません。 |
|  `metadata` |  ユーザー定義のアーティファクトメタデータ。アーティファクトに関連する構造化データ。 |
|  `name` |  二次（ポートフォリオ）コレクションのアーティファクト名とバージョン。形式は{collection}:{alias}。名前だけが含まれます。 |
|  `project` |  二次（ポートフォリオ）アーティファクトコレクションのプロジェクト名。 |
|  `qualified_name` |  二次（ポートフォリオ）コレクションのentity/project/name。 |
|  `size` |  アーティファクトの総サイズ（バイト単位）。 |
|  `source_collection` |  アーティファクトの一次（シーケンス）コレクション。 |
|  `source_entity` |  一次（シーケンス）アーティファクトコレクションのエンティティの名前。 |
|  `source_name` |  一次（シーケンス）コレクションのアーティファクト名とバージョン。形式は {collection}:{alias} です。名前だけが含まれます。 |
|  `source_project` |  一次（シーケンス）アーティファクトコレクションのプロジェクト名。 |
|  `source_qualified_name` |  一次（シーケンス）コレクションのentity/project/name。 |
|  `source_version` |  一次（シーケンス）コレクションのアーティファクトのバージョン。形式は "v{number}"。 |
|  `state` |  アーティファクトの状態。 "PENDING", "COMMITTED", または "DELETED" のいずれか。 |
|  `ttl` |  アーティファクトのタイム・トゥ・リブ (TTL) ポリシー。 |
|  `type` |  アーティファクトのタイプ。一般的なタイプには`dataset`や`model`が含まれます。 |
|  `updated_at` |  アーティファクトが最後に更新された時刻。 |
|  `version` |  二次（ポートフォリオ）コレクションのアーティファクトのバージョン。 |

## Methods

### `add`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1344-L1441)

```python
add(
    obj: data_types.WBValue,
    name: StrPath
) -> ArtifactManifestEntry
```

wandb.WBValue `obj`をアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `obj` |  追加するオブジェクト。現在サポートされているのは、Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3Dです。 |
|  `name` |  アーティファクト内でオブジェクトを追加するパス。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |

### `add_dir`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1200-L1260)

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
|  `local_path` |  ローカルディレクトリのパス。 |
|  `name` |  アーティファクト内のサブディレクトリ名。この名前はW&BアプリのUIでアーティファクトの`type`によってネストされて表示されます。デフォルトはアーティファクトのルートです。 |
|  `skip_cache` |  `True`に設定されている場合、アップロード中にファイルをキャッシュにコピー/移動しません。 |
|  `policy` |  "mutable" または "immutable"。デフォルトは"mutable"。「mutable」：アップロード中のファイルの破損を防ぐため、一時コピーを作成します。「immutable」：保護を無効にし、ファイルを削除または変更しないようユーザーに依存します。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |
|  `ValueError` |  ポリシーは "mutable" または "immutable" でなければなりません。 |

### `add_file`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1154-L1198)

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
|  `local_path` |  追加するファイルのパス。 |
|  `name` |  追加するファイルが使用されるアーティファクト内のパス。デフォルトはファイルのベース名です。 |
|  `is_tmp` |  trueの場合、ファイルは衝突を避けるため決定論的に名前が変更されます。 |
|  `skip_cache` |  `True`に設定されている場合、アップロード後にファイルをキャッシュにコピーしません。 |
|  `policy` |  "mutable" または "immutable"。デフォルトは"mutable"。「mutable」：アップロード中のファイルの破損を防ぐため、一時コピーを作成します。「immutable」：保護を無効にし、ファイルを削除または変更しないようユーザーに依存します。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |
|  `ValueError` |  ポリシーは "mutable" または "immutable" でなければなりません。 |

### `add_reference`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1262-L1342)

```python
add_reference(
    uri: Union[ArtifactManifestEntry, str],
    name: Optional[StrPath] = None,
    checksum: bool = (True),
    max_objects: Optional[int] = None
) -> Sequence[ArtifactManifestEntry]
```

URIで示される参照をアーティファクトに追加します。

ファイルやディレクトリをアーティファクトに追加するのとは異なり、参照はW&Bにアップロードされません。詳細については、[外部ファイルの追跡](https://docs.wandb.ai/guides/artifacts/track-external-files)を参照してください。

デフォルトでは、次のスキームがサポートされています：

- http(s): ファイルのサイズとダイジェストは、サーバーが返す `Content-Length` と `ETag` 応答ヘッダーによって推測されます。
- s3: チェックサムとサイズはオブジェクトのメタデータから取得されます。バケットバージョニングが有効になっている場合、バージョンIDも追跡されます。
- gs: チェックサムとサイズはオブジェクトのメタデータから取得されます。バケットバージョニングが有効になっている場合、バージョンIDも追跡されます。
- https, ドメインが `*.blob.core.windows.net` (Azure) と一致する場合: ブロブのメタデータからチェックサムとサイズが取得されます。ストレージアカウントのバージョニングが有効になっている場合、バージョンIDも追跡されます。
- file: チェックサムとサイズはファイルシステムから取得されます。このスキームは、NFSシェアまたは他の外部マウントボリュームに含まれるファイルを追跡するのに便利ですが、必ずしもアップロードする必要はありません。

他のスキームの場合、ダイジェストはURIのハッシュであり、サイズは空白のままです。

| 引数 |  |
| :--- | :--- |
|  `uri` |  追加する参照のURIパス。URIパスは、他のアーティファクトのエントリへの参照を保存するために`Artifact.get_entry`から返されるオブジェクトである可能性があります。 |
|  `name` |  この参照の内容を配置するアーティファクト内のパス。 |
|  `checksum` |  参照URIにあるリソースのチェックサムを行うかどうか。チェックサミングは強く推奨されます。 |
|  `max_objects` |  ディレクトリまたはバケットストアプレフィックスを指す参照を追加する際に考慮するオブジェクトの最大数。デフォルトでは、Amazon S3, GCS, Azure, およびローカルファイルの最大数は10,000,000です。他のURIスキーマには最大値がありません。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |

### `checkout`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1878-L1907)

```python
checkout(
    root: Optional[str] = None
) -> str
```

指定されたルートディレクトリをアーティファクトの内容で置き換えます。

警告：これはアーティファクトに含まれていない`root`内のすべてのファイルを削除します。

| 引数 |  |
| :--- | :--- |
|  `root` |  このアーティファクトのファイルで置き換えるディレクトリ。 |

| 戻り値 |  |
| :--- | :--- |
|  チェックアウトされた内容のパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `delete`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2019-L2038)

```python
delete(
    delete_aliases: bool = (False)
) -> None
```

アーティファクトとそのファイルを削除します。

リンクされたアーティファクト（つまり、ポートフォリオコレクションのメンバー）に対して呼び出された場合、リンクのみが削除され、元のアーティファクトは影響を受けません。

| 引数 |  |
| :--- | :--- |
|  `delete_aliases` |  `True`に設定すると、アーティファクトに関連付けられているすべてのエイリアスを削除します。そうでない場合、既存のエイリアスが存在する場合に例外が発生します。このパラメータは、アーティファクトがリンクされている場合（つまり、ポートフォリオコレクションのメンバーである場合）には無視されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `download`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1621-L1662)

```python
download(
    root: Optional[StrPath] = None,
    allow_missing_references: bool = (False),
    skip_cache: Optional[bool] = None,
    path_prefix: Optional[StrPath] = None
) -> FilePathStr
```

アーティファクトの内容を指定されたルートディレクトリにダウンロードします。

`root`内の既存のファイルは変更されません。一致するようにしたい場合は、`download`を呼び出す前に明示的に`root`を削除してください。

| 引数 |  |
| :--- | :--- |
|  `root` |  W&Bがアーティファクトのファイルを保存するディレクトリ。 |
|  `allow_missing_references` |  `True`に設定すると、無効な参照パスがあっても無視され、参照ファイルのダウンロードは行われません。 |
|  `skip_cache` |  `True`に設定されている場合、アーティファクトキャッシュをスキップし、各ファイルをデフォルトのルートまたは指定されたダウンロードディレクトリに直接ダウンロードします。 |

| 戻り値 |  |
| :--- | :--- |
|  ダウンロードされた内容のパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `file`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1950-L1975)

```python
file(
    root: Optional[str] = None
) -> StrPath
```

単一のファイルアーティファクトを、`root`で指定されたディレクトリにダウンロードします。

| 引数 |  |
| :--- | :--- |
|  `root` |  ファイルを保存するルートディレクトリ。デフォルトは `./artifacts/self.name/`。 |

| 戻り値 |  |
| :--- | :--- |
|  ダウンロードされたファイルのフルパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |
|  `ValueError` |  アーティファクトに複数のファイルが含まれている場合。 |

### `files`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1977-L1994)

```python
files(
    names: Optional[List[str]] = None,
    per_page: int = 50
) -> ArtifactFiles
```

このアーティファクトに保存されているすべてのファイルを反復します。

| 引数 |  |
| :--- | :--- |
|  `names` |  リストしたいアーティファクトのルート相対パスのファイル名。 |
|  `per_page` |  リクエストごとに返されるファイル数。 |

| 戻り値 |  |
| :--- | :--- |
|  `File`オブジェクトを含むイテレータ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `finalize`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L723-L731)

```python
finalize() -> None
```

アーティファクトバージョンを確定します。

アーティファクトバージョンが確定されると、アーティファクトに変更を加えることはできません。アーティファクトにデータを追加する場合は、新しいアーティファクトバージョンを作成してください。アーティファクトは`log_artifact`でログに記録されると自動的に確定されます。

### `get`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1537-L1583)

```python
get(
    name: str
) -> Optional[data_types.WBValue]
```

アーティファクト相対の `name` にあるWBValueオブジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクト相対の名前。 |

| 戻り値 |  |
| :--- | :--- |
|  `W&B`オブジェクト。`wandb.log()`でログに記録し、W&BのUIで可視化可能。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合やrunがオフラインの場合。 |

### `get_added_local_path_name`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1585-L1597)

```python
get_added_local_path_name(
    local_path: str
) -> Optional[str]
```

ローカルファイルシステムパスによって追加されたファイルのアーティファクト相対名を取得します。

| 引数 |  |
| :--- | :--- |
|  `local_path` |  アーティファクト相対名に解決されるローカルパス。 |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクト相対名。 |

### `get_entry`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1515-L1535)

```python
get_entry(
    name: StrPath
) -> ArtifactManifestEntry
```

指定された名前でエントリを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクト相対の名前。 |

| 戻り値 |  |
| :--- | :--- |
|  `W&B`オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合やrunがオフラインの場合。 |
|  `KeyError` |  指定された名前のエントリがアーティファクトに含まれていない場合。 |

### `get_path`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1507-L1513)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

非推奨。`get_entry(name)`を使用してください。

### `is_draft`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L741-L746)

```python
is_draft() -> bool
```

アーティファクトが保存されていないかどうかを確認します。

戻り値: Boolean。アーティファクトが保存されている場合は`False`。保存されていない場合は`True`。

### `json_encode`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2229-L2236)

```python
json_encode() -> Dict[str, Any]
```

アーティファクトをJSON形式にエンコードして返します。

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトの属性を表す文字列キーを持つ`dict`。 |

### `link`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2065-L2093)

```python
link(
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

このアーティファクトをポートフォリオ（アーティファクトのプロモートされたコレクション）にリンクします。

| 引数 |  |
| :--- | :--- |
|  `target_path` |  プロジェクト内のポートフォリオへのパス。ターゲットパスは以下のいずれかのスキーマに従う必要があります `{portfolio}`, `{project}/{portfolio}` または `{entity}/{project}/{portfolio}`。モデルレジストリへアーティファクトをリンクする場合は、スキーマ `{"model-registry"}/{Registered Model Name}` または `{entity}/{"model-registry"}/{Registered Model Name}`を設定してください。 |
|  `aliases` |  指定されたポートフォリオ内でアーティファクトを一意に識別する文字列のリスト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `logged_by`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2184-L2227)

```python
logged_by() -> Optional[Run]
```

アーティファクトを最初にログに記録したW&B runを取得します。

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトを最初にログに記録したW&B runの名前。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `new_draft`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L345-L377)

```python
new_draft() -> "Artifact"
```

このコミットされたアーティファクトと同じ内容を持つ新しい下書きアーティファクトを作成します。

戻り値のアーティファクトは拡張や変更が可能で、新しいバージョンとしてログに記録できます。

| 戻り値 |  |
| :--- | :--- |
|  `Artifact`オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `new_file`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1115-L1152)

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
|  `encoding` |  新しいファイルを開くために使用されるエンコーディング。 |

| 戻り値 |  |
| :--- | :--- |
|  新しいファイルオブジェクト。クローズするとファイルは自動的にアーティファクトに追加されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |

### `path_contains_dir_prefix`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1664-L1671)

```python
@classmethod
path_contains_dir_prefix(
    path: StrPath,
    dir_path: StrPath
) -> bool
```

`path`が`dir_path`をプレフィックスとして含む場合にtrueを返します。

### `remove`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1476-L1505)

```python
remove(
    item: Union[StrPath, 'ArtifactManifestEntry']
) -> None
```

アーティファクトからアイテムを削除します。

| 引数 |  |
| :--- | :--- |
|  `item` |  削除するアイテム。特定のマニフェストエントリまたはアーティファクト相対パス。アイテムがディレクトリと一致する場合、そのディレクトリ内のすべてのアイテムが削除されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |
|  `FileNotFoundError` |  アイテムがアーティファクトに見つからない場合。 |

### `save`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L751-L790)

```python
save(
    project: Optional[str] = None,
    settings: Optional['wandb.wandb_sdk.wandb_settings.Settings'] = None
) -> None
```

アーティファクトに対する変更を保存します。

現在のrun内にある場合、そのrunでこのアーティファクトがログに記録されます。run内にない場合、"auto"タイプのrunが作成され、このアーティファクトを追跡します。

| 引数 |  |
| :--- | :--- |
|  `project` |  runが既にコンテキストにない場合にアーティファクトに使用するプロジェクト。 |
|  `settings` |  自動runを初期化するときに使用する設定オブジェクト。主にテスト環境で使用されます。 |

### `should_download_entry`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1673-L1679)

```python
@classmethod
should_download_entry(
    entry: ArtifactManifestEntry,
    prefix: Optional[StrPath]
) -> bool
```

### `unlink`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2095-L2111)

```python
unlink() -> None
```

現在、ポートフォリオ（アーティファクトのプロモートされたコレクション）のメンバーである場合、このアーティファクトのリンクを解除します。

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |
|  `ValueError` |  アーティファクトがリンクされていない場合、つまりポートフォリオコレクションのメンバーでない場合。 |

### `used_by`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2137-L2182)

```python
used_by() -> List[Run]
```

このアーティファクトを使用したrunのリストを取得します。

| 戻り値 |  |
| :--- | :--- |
|  `Run`オブジェクトのリスト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `verify`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1909-L1948)

```python
verify(
    root: Optional[str] = None
) -> None
```

アーティファクトの内容がマニフェストと一致するかどうかを検証します。

ディレクトリ内のすべてのファイルがチェックサムされ、チェックサムはアーティファクトのマニフェストとクロスリファレンスされます。参照は検証されません。

| 引数 |  |
| :--- | :--- |
|  `root` |  検証するディレクトリ。Noneの場合、アーティファクトは `./artifacts/self.name/` にダウンロードされます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |
|  `ValueError` |  検証に失敗した場合。 |

### `wait`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L798-L819)

```python
wait(
    timeout: Optional[int] = None
) -> "Artifact"
```

必要に応じて、このアーティファクトのロギングが完了するまで待ちます。

| 引数 |  |
| :--- | :--- |
|  `timeout` |  待機する時間（秒）。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact`オブジェクト。 |

### `__getitem__`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1085-L1097)

```python
__getitem__(
    name: str
) -> Optional[data_types.WBValue]
```

アーティファクト相対の `name` にあるWBValueオブジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクト相対の名前。 |

| 戻り値 |  |
| :--- | :--- |
|  W&Bオブジェクト。`wandb.log()`でログに記録し、W&BのUIで可視化できます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合やrunがオフラインの場合。 |

### `__setitem__`

[ソースを見る](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1099-L1113)

```python
__setitem__(
    name: str,
    item: data_types.WBValue
) -> ArtifactManifestEntry
```

`item`を指定されたパス `name` のアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  オブジェクトを追加するアーティファクト内のパス。 |
|  `item` |  追加するオブジェクト。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |