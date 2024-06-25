
# Artifact

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L90-L2349' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

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

空の W&B Artifact を構築します。`add` で始まるメソッドでアーティファクトの内容を埋めます。すべての必要なファイルがアーティファクトに追加されたら、`wandb.log_artifact()` を呼んでログを取ります。

| 引数 |  |
| :--- | :--- |
|  `name` |  アーティファクトの人間が読める名前。W&BアプリのUIまたはプログラムで特定のアーティファクトを識別するために名前を使用します。アーティファクトを対話的に参照するには `use_artifact` のパブリックAPIを使用します。名前には文字、数字、アンダースコア、ハイフン、ドットを含めることができます。名前はプロジェクト内で一意である必要があります。 |
|  `type` |  アーティファクトのタイプ。アーティファクトを整理し、区別するためにタイプを使用します。文字、数字、アンダースコア、ハイフン、ドットを含める任意の文字列を使用できます。一般的なタイプには `dataset` または `model` があります。アーティファクトをW&Bモデルレジストリにリンクする場合、タイプ文字列に `model` を含めます。 |
|  `description` |  アーティファクトの説明。Model または Dataset Artifacts の場合、標準化されたチームモデルまたはデータセットカードのドキュメントを追加します。アーティファクトの説明をプログラムで表示するには、`Artifact.description` 属性または W&BアプリのUIを使用します。W&Bは説明をマークダウンとしてレンダリングします。 |
|  `metadata` |  アーティファクトに関する追加情報。メタデータはキーと値のペアの辞書として指定します。合計で100個以下のキーを指定できます。 |

| 返り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

| 属性 |  |
| :--- | :--- |
|  `aliases` |  アーティファクトバージョンに割り当てられた1つ以上の意味的にフレンドリーな参照または識別「ニックネーム」のリスト。エイリアスはプログラムで参照できる可変参照です。アーティファクトのエイリアスを W&BアプリのUIやプログラムで変更できます。詳細は [Create new artifact versions](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version) を参照してください。 |
|  `collection` |  このアーティファクトが取得されたコレクション。コレクションはアーティファクトバージョンの順序付けられたグループです。このアーティファクトがポートフォリオ/リンクコレクションから取得された場合、そのコレクションが返され、アーティファクトバージョンが由来したコレクションではなくなります。アーティファクトが発信元となるコレクションはソースシーケンスとして知られています。 |
|  `commit_hash` |  このアーティファクトがコミットされた時に返されるハッシュ。 |
|  `created_at` |  アーティファクトが作成されたタイムスタンプ。 |
|  `description` |  アーティファクトの説明。 |
|  `digest` |  アーティファクトの論理ダイジェスト。ダイジェストはアーティファクトの内容のチェックサムです。アーティファクトが現在の `latest` バージョンと同じダイジェストを持っている場合、`log_artifact` は何も行いません。 |
|  `entity` |  二次（ポートフォリオ）アーティファクトコレクションのエンティティの名前。 |
|  `file_count` |  ファイルの数（リファレンスを含む）。 |
|  `id` |  アーティファクト ID。 |
|  `manifest` |  アーティファクトのマニフェスト。マニフェストにはすべての内容が記載されており、アーティファクトがログに記録された後は変更できません。 |
|  `metadata` |  ユーザー定義のアーティファクトメタデータ。アーティファクトに関連する構造化データ。 |
|  `name` |  二次（ポートフォリオ）コレクション内のアーティファクト名およびバージョン。形式 {collection}:{alias} の文字列。アーティファクトが保存される前は、バージョンがまだ知られていないため、名前のみが含まれています。 |
|  `project` |  二次（ポートフォリオ）アーティファクトコレクションのプロジェクトの名前。 |
|  `qualified_name` |  二次（ポートフォリオ）コレクションの entity/project/name。 |
|  `size` |  バイト単位のアーティファクトの総サイズ。このアーティファクトが追跡する任意のリファレンスを含みます。 |
|  `source_collection` |  アーティファクトの一次（シーケンス）コレクション。 |
|  `source_entity` |  一次（シーケンス）アーティファクトコレクションのエンティティの名前。 |
|  `source_name` |  一次（シーケンス）コレクション内のアーティファクト名およびバージョン。形式 {collection}:{alias} の文字列。アーティファクトが保存される前は、バージョンがまだ知られていないため、名前のみが含まれています。 |
|  `source_project` |  一次（シーケンス）アーティファクトコレクションのプロジェクト名。 |
|  `source_qualified_name` |  一次（シーケンス）コレクションの entity/project/name。 |
|  `source_version` |  一次（シーケンス）コレクション内のアーティファクトのバージョン。形式 "v{number}" の文字列。 |
|  `state` |  アーティファクトのステータス。次のいずれか: "PENDING", "COMMITTED", "DELETED"。 |
|  `ttl` |  アーティファクトのタイム・トゥ・リブ（TTL）ポリシー。TTLポリシーの期間が過ぎるとすぐにアーティファクトは削除されます。`None` に設定すると、アーティファクトはTTLポリシーを無効にし、チームデフォルトのTTLが存在しても削除対象にはなりません。チーム管理者がデフォルトTTLを定義し、アーティファクトにカスタムポリシーが設定されていない場合、アーティファクトはチームデフォルトのTTLポリシーを継承します。 |
|  `type` |  アーティファクトのタイプ。一般的なタイプには `dataset` または `model` があります。 |
|  `updated_at` |  アーティファクトが最後に更新された時間。 |
|  `version` |  二次（ポートフォリオ）コレクション内のアーティファクトのバージョン。 |

## メソッド

### `add`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1344-L1441)

```python
add(
    obj: data_types.WBValue,
    name: StrPath
) -> ArtifactManifestEntry
```

wandb.WBValue オブジェクトをアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `obj` |  追加するオブジェクト。現在、Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3D のいずれかをサポートしています。 |
|  `name` |  オブジェクトを追加するアーティファクト内のパス。 |

| 返り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンは確定されているため、変更はできません。新しいアーティファクトバージョンをログに記録します。 |

### `add_dir`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1200-L1260)

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
|  `name` |  アーティファクト内のサブディレクトリ名。指定した名前はW&BアプリのUIにアーティファクトの `type` によってネストされて表示されます。デフォルトはアーティファクトのルートです。 |
|  `skip_cache` |  `True` に設定すると、アップロード中にファイルをキャッシュにコピー/移動しません。 |
|  `policy` |  "mutable" | "immutable"。デフォルトは "mutable"。"mutable": アップロード中のファイルの破損を防ぐために一時コピーを作成します。"immutable": 保護を無効にし、ユーザーがファイルを削除または変更しないようにします。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンは確定されているため、変更はできません。新しいアーティファクトバージョンをログに記録します。 |
|  `ValueError` |  ポリシーは "mutable" または "immutable" でなければなりません。 |

### `add_file`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1154-L1198)

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
|  `name` |  追加されるファイルのアーティファクト内のパス。デフォルトはファイルのベースネームです。 |
|  `is_tmp` |  `True` の場合、競合を避けるためファイル名が決定的に変更されます。 |
|  `skip_cache` |  `True` に設定すると、アップロード後にファイルをキャッシュにコピーしません。 |
|  `policy` |  "mutable" | "immutable"。デフォルトは "mutable"。"mutable": アップロード中のファイルの破損を防ぐために一時コピーを作成します。"immutable": 保護を無効にし、ユーザーがファイルを削除または変更しないようにします。 |

| 返り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンは確定されているため、変更はできません。新しいアーティファクトバージョンを記録します。 |
|  `ValueError` |  ポリシーは "mutable" または "immutable" でなければなりません。 |

### `add_reference`

[View source](https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/artifacts/artifact.py#L1262-L1342)

```python
add_reference(
    uri: Union[ArtifactManifestEntry, str],
    name: Optional[StrPath] = None,
    checksum: bool = (True),
    max_objects: Optional[int] = None
) -> Sequence[ArtifactManifestEntry]
```

URIで示されるリファレンスをアーティファクトに追加します。

ファイルやディレクトリをアーティファクトに追加する場合と異なり、リファレンスは W&B にアップロードされません。詳細は
[外部ファイルの追跡](https://docs.wandb.ai/guides/artifacts/track-external-files) を参照してください。

デフォルトでは、以下のスキームがサポートされています：

- http(s): ファイルのサイズとダイジェストは、サーバーが返す `Content-Length` と `ETag` レスポンスヘッダーによって推論されます。
- s3: チェックサムとサイズはオブジェクトメタデータから取得されます。バケットバージョン管理が有効であれば、バージョンIDも追跡されます。
- gs: チェックサムとサイズはオブジェクトメタデータから取得されます。バケットバージョン管理が有効であれば、バージョンIDも追跡されます。
- https、`*.blob.core.windows.net` と一致するドメイン（Azure）: チェックサムとサイズはblobメタデータから取得され、ストレージアカウントバージョン管理が有効であれば、バージョンIDも追跡されます。
- file: チェックサムとサイズはファイルシステムから取得されます。このスキームは、NFS共有や他の外部マウントボリュームに含まれるファイルを追跡するのに役立ちますが、必ずしもアップロードする必要はありません。

他のスキームについては、ダイジェストはURIのハッシュであり、サイズは空白のままです。

| 引数 |  |
| :--- | :--- |
|  `uri` |  追加するリファレンスのURIパス。URIパスは `Artifact.get_entry` から返されたオブジェクトで、他のアーティファクトのエントリへのリファレンスを格納することができます。 |
|  `name` |  このリファレンスの内容を配置するアーティファクト内のパス。 |
|  `checksum` |  リファレンスURIにあるリソースのチェックサムを計算するかどうか。チェックサムは、自動的な整合性検証を可能にするため、強く推奨されます。チェックサムを無効にすると、アーティファクトの作成が速くなりますが、リファレンスディレクトリは反復されず、ディレクトリのオブジェクトはアーティファクトに保存されなくなります。チェックサムが無効な場合は、リファレンスオブジェクトを追加することをお勧めします。 |
|  `max_objects` |  ディレクトリやバケットストアプレフィックスを指すリファレンスを追加する場合に考慮される最大オブジェクト数。デフォルトで、Amazon S3、GCS、Azure、およびローカルファイルの最大オブジェクト数は10,000,000です。他のURIスキーマには最大値はありません。 |

| 返り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリのシーケンス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現