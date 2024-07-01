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

空の W&B Artifact を構築します。メソッドでアーティファクトの内容を追加します。すべてのファイルが追加されたら、`wandb.log_artifact()`を呼び出してログを記録します。

| 引数 |  |
| :--- | :--- |
|  `name` |  アーティファクトの人間が読みやすい名前。W&B アプリ UI またはプログラムで特定のアーティファクトを識別するために名前を使用します。 `use_artifact` パブリック API を使用してインタラクティブにアーティファクトを参照できます。名前は文字、数字、アンダースコア、ハイフン、およびドットを含むことができます。名前はプロジェクト内で一意である必要があります。 |
|  `type` |  アーティファクトのタイプ。アーティファクトのタイプを使用して、アーティファクトを整理および区別します。任意の文字列を使用できますが、文字、数字、アンダースコア、ハイフン、およびドットを含む必要があります。一般的なタイプには `dataset` や `model` があります。アーティファクトを W&B Model Registry にリンクする場合は、タイプ文字列に `model` を含めます。 |
|  `description` |  アーティファクトの説明。Model や Dataset Artifacts の場合、標準化されたチームモデルやデータセットカードのドキュメントを追加します。アーティファクトの説明は `Artifact.description` 属性を使用してプログラム的に参照したり、W&B アプリ UI を使用して参照できます。W&B は W&B アプリ内でマークダウンとして説明をレンダリングします。 |
|  `metadata` |  アーティファクトに関する追加情報。キーと値のペアの辞書としてメタデータを指定します。合計 100 個のキーを超えることはできません。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

| 属性 |  |
| :--- | :--- |
|  `aliases` |  アーティファクトバージョンに割り当てられた、セマンティックにフレンドリーな参照または識別「ニックネーム」のリスト。エイリアスは、プログラム的に参照できる可変の参照です。エイリアスは、W&B アプリ UI またはプログラム的に変更できます。詳細については、[新しいアーティファクトバージョンの作成](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version) を参照してください。 |
|  `collection` |  このアーティファクトが取得されたコレクション。コレクションはアーティファクトバージョンの順序グループです。このアーティファクトがポートフォリオ/リンクコレクションから取得された場合、そのコレクションが返されます。このアーティファクトバージョンが発生したコレクションはソースシーケンスと呼ばれます。 |
|  `commit_hash` |  このアーティファクトがコミットされたときに返されるハッシュ。 |
|  `created_at` |  アーティファクトが作成されたときのタイムスタンプ。 |
|  `description` |  アーティファクトの説明。 |
|  `digest` |  アーティファクトの論理ダイジェスト。ダイジェストはアーティファクト内容のチェックサムです。アーティファクトが現在の最新バージョンと同じダイジェストを持っている場合、`log_artifact`は何もしません。 |
|  `entity` |  セカンダリー（ポートフォリオ）アーティファクトコレクションのエンティティ名。 |
|  `file_count` |  ファイルの数（参照を含む）。 |
|  `id` |  アーティファクトのID。 |
|  `manifest` |  アーティファクトのマニフェスト。マニフェストにはすべての内容が一覧されており、アーティファクトがログに記録されると変更できません。 |
|  `metadata` |  ユーザー定義のアーティファクトメタデータ。アーティファクトに関連付けられた構造化データ。 |
|  `name` |  セカンダリー（ポートフォリオ）コレクションのアーティファクト名とバージョン。形式は {collection}:{alias} の文字列です。アーティファクトが保存されるまで、バージョンはまだ知られていないため、名前のみを含みます。 |
|  `project` |  セカンダリー（ポートフォリオ）アーティファクトコレクションのプロジェクト名。 |
|  `qualified_name` |  セカンダリー（ポートフォリオ）コレクションの entity/project/name。 |
|  `size` |  バイト単位のアーティファクトの総サイズ。このアーティファクトが追跡する参照を含みます。 |
|  `source_collection` |  アーティファクトのプライマリー（シーケンス）コレクション。 |
|  `source_entity` |  プライマリー（シーケンス）アーティファクトコレクションのエンティティ名。 |
|  `source_name` |  プライマリー（シーケンス）コレクションのアーティファクト名とバージョン。形式は {collection}:{alias} の文字列です。アーティファクトが保存されるまで、名前のみを含みます。 |
|  `source_project` |  プライマリー（シーケンス）アーティファクトコレクションのプロジェクト名。 |
|  `source_qualified_name` |  プライマリー（シーケンス）コレクションの entity/project/name。 |
|  `source_version` |  プライマリー（シーケンス）コレクションのアーティファクトのバージョン。形式は "v{number}" の文字列です。 |
|  `state` |  アーティファクトの状態。次のいずれか："PENDING", "COMMITTED", または "DELETED". |
|  `ttl` |  アーティファクトのデータ保存期間（TTL）ポリシー。TTL ポリシーの期間が終了すると、アーティファクトは間もなく削除されます。`None` に設定すると、アーティファクトは TTL ポリシーを無効にし、チームデフォルト TTL が存在しても削除されません。アーティファクトは、チーム管理者がデフォルト TTL を定義し、アーティファクトにカスタムポリシーが設定されていない場合、チームデフォルトから TTL ポリシーを継承します。 |
|  `type` |  アーティファクトのタイプ。一般的なタイプには `dataset` や `model` があります。 |
|  `updated_at` |  アーティファクトが最後に更新された時間。 |
|  `version` |  セカンダリー（ポートフォリオ）コレクションのアーティファクトのバージョン。 |

## メソッド

### `add`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1344-L1441)

```python
add(
    obj: data_types.WBValue,
    name: StrPath
) -> ArtifactManifestEntry
```

wandb.WBValue `obj` をアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `obj` |  追加するオブジェクト。現在サポートされているのは Bokeh、JoinedTable、PartitionedTable、Table、Classes、ImageMask、BoundingBoxes2D、Audio、Image、Video、Html、Object3D のいずれか。 |
|  `name` |  オブジェクトを追加するアーティファクト内のパス。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |

### `add_dir`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1200-L1260)

```python
add_dir(
    local_path: str,
    name: Optional[str] = None,
    skip_cache: Optional[bool] = (False),
    policy: Optional[Literal['mutable', 'immutable']] = "mutable"
) -> None
```

ローカルディレクトリーをアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `local_path` |  ローカルディレクトリーのパス。 |
|  `name` |  アーティファクト内のサブディレクトリー名。W&B アプリ UI に表示される名前。デフォルトはアーティファクトのルート。 |
|  `skip_cache` |  `True` に設定した場合、アップロード時にファイルをキャッシュにコピー/移動しません。 |
|  `policy` |  "mutable" または "immutable"。デフォルトは "mutable" "mutable"。一時コピーを作成してアップロード中の破損を防ぎます。"immutable"：保護を無効にし、ユーザーがファイルを削除または変更しないことを前提とします。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |
|  `ValueError` |  ポリシーは "mutable" または "immutable" でなければなりません。 |

### `add_file`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1154-L1198)

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
|  `name` |  アーティファクト内でファイルに使用するパス。デフォルトはファイルのベース名。 |
|  `is_tmp` |  true の場合、競合を回避するためにファイルの名前が決定論的に変更されます。 |
|  `skip_cache` |  `True` に設定した場合、アップロード後にファイルをキャッシュにコピーしません。 |
|  `policy` |  "mutable" または "immutable"。デフォルトは "mutable" "mutable"。一時コピーを作成してアップロード中の破損を防ぎます。"immutable"：保護を無効にし、ユーザーがファイルを削除または変更しないことを前提とします。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |
|  `ValueError` |  ポリシーは "mutable" または "immutable" でなければなりません。 |

### `add_reference`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1262-L1342)

```python
add_reference(
    uri: Union[ArtifactManifestEntry, str],
    name: Optional[StrPath] = None,
    checksum: bool = (True),
    max_objects: Optional[int] = None
) -> Sequence[ArtifactManifestEntry]
```

URI で指定される参照をアーティファクトに追加します。

ファイルまたはディレクトリーをアーティファクトに追加するのとは異なり、参照は W&B にアップロードされません。 詳細については、[外部ファイルの追跡](https://docs.wandb.ai/guides/artifacts/track-external-files) を参照してください。

デフォルトでは、以下のスキームがサポートされています：

- http(s): サーバーが返す `Content-Length` および `ETag` 応答ヘッダーによって、ファイルのサイズとダイジェストが推測されます。
- s3: チェックサムとサイズはオブジェクトメタデータから取得されます。バケットのバージョン管理が有効になっている場合は、バージョン ID も追跡されます。
- gs: チェックサムとサイズはオブジェクトメタデータから取得されます。バケットのバージョン管理が有効になっている場合は、バージョン ID も追跡されます。
- https, *.blob.core.windows.net ドメイン一致 (Azure): チェックサムとサイズが Blob メタデータから取得されます。ストレージアカウントのバージョン管理が有効になっている場合は、バージョン ID も追跡されます。
- file: チェックサムとサイズがファイルシステムから取得されます。このスキームは、 NFS シェアや他の外部マウントボリュームを追跡するが、アップロードしない場合に便利です。

その他のスキームについては、ダイジェストは URI のハッシュであり、サイズは空白のままです。

| 引数 |  |
| :--- | :--- |
|  `uri` |  参照を追加する URI パス。URI パスは、他のアーティファクトのエントリへの参照を保存するための `Artifact.get_entry` から返されるオブジェクトであることもあります。 |
|  `name` |  この参照の内容を配置するアーティファクト内のパス。 |
|  `checksum` |  参照 URI にあるリソースのチェックサムを取るかどうか。チェックサムは自動整合性検証を可能にするため、強く推奨されます。チェックサムを無効にすると、アーティファクトの作成速度が向上しますが、参照ディレクトリーは反復されないため、ディレクトリー内のオブジェクトはアーティファクトに保存されません。チェックサムが false の場合は、参照オブジェクトを追加することをお勧めします。 |
|  `max_objects` |  ディレクトリーまたはバケットストアプレフィックスを指す参照を追加するときに考慮する最大オブジェクト数。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |

### `checkout`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1878-L1907)

```python
checkout(
    root: Optional[str] = None
) -> str
```

指定されたルートディレクトリをアーティファクトの内容で置き換える。

警告: これはアーティファクトに含まれていない `root` のすべてのファイルを削除します。

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

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2019-L2038)

```python
delete(
    delete_aliases: bool = (False)
) -> None
```

アーティファクトとそのファイルを削除します。

リンクされたアーティファクトで呼び出された場合（例：ポートフォリオコレクションのメンバーの場合）、リンクのみが削除され、ソースアーティファクトには影響しません。

| 引数 |  |
| :--- | :--- |
|  `delete_aliases` |  `True` に設定した場合、アーティファクトに関連するすべてのエイリアスを削除します。それ以外の場合、アーティファクトに既存のエイリアスがある場合は例外を発生させます。このパラメータは、アーティファクトがリンクされている場合（例：ポートフォリオコレクションのメンバーである場合）には無視されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `download`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1621-L1662)

```python
download(
    root: Optional[StrPath] = None,
    allow_missing_references: bool = (False),
    skip_cache: Optional[bool] = None,
    path_prefix: Optional[StrPath] = None
) -> FilePathStr
```

アーティファクトの内容を指定されたルートディレクトリにダウンロードします。

既存の `root` 内のファイルは変更されません。`download` を呼び出す前に `root` を明示的に削除して、`root` の内容がアーティファクトと完全に一致するようにします。

| 引数 |  |
| :--- | :--- |
|  `root` |  アーティファクトのファイルを保存するディレクトリ。 |
|  `allow_missing_references` |  `True` に設定した場合、無効な参照パスを無視して参照ファイルをダウンロードします。 |
|  `skip_cache` |  `True` に設定した場合、ダウンロード時にアーティファクトキャッシュをスキップし、デフォルトのルートまたは指定されたダウンロードディレクトリにファイルをダウンロードします。 |

| 戻り値 |  |
| :--- | :--- |
|  ダウンロードされた内容のパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `file`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1950-L1975)

```python
file(
    root: Optional[str] = None
) -> StrPath
```

指定された `root` に単一のファイルアーティファクトをダウンロードします。

| 引数 |  |
| :--- | :--- |
|  `root` |  ファイルを保存するルートディレクトリ。デフォルトは './artifacts/self.name/'。 |

| 戻り値 |  |
| :--- | :--- |
|  ダウンロードされたファイルのフルパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |
|  `ValueError` |  アーティファクトに複数のファイルが含まれている場合。 |

### `files`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1977-L1994)

```python
files(
    names: Optional[List[str]] = None,
    per_page: int = 50
) -> ArtifactFiles
```

このアーティファクトに保存されているすべてのファイルを繰り返し処理します。

| 引数 |  |
| :--- | :--- |
|  `names` |  リストするアーティファクトのルート相対パス。 |
|  `per_page` |  1 リクエストあたりのファイル数。 |

| 戻り値 |  |
| :--- | :--- |
|  `File` オブジェクトを含む反復子。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `finalize`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L723-L731)

```python
finalize() -> None
```

アーティファクトバージョンを確定します。

アーティファクトバージョンが確定されると、そのバージョンには変更を加えることができません。それ以降のデータをログに記録するためには、新しいアーティファクトバージョンを作成する必要があります。アーティファクトは `log_artifact` を使用してログに記録されると自動的に確定されます。

### `get`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1537-L1583)

```python
get(
    name: str
) -> Optional[data_types.WBValue]
```

アーティファクト相対 `name` に配置された WBValue オブジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクト相対名。 |

| 戻り値 |  |
| :--- | :--- |
|  W&B オブジェクト。これは `wandb.log()` を使用してログに記録され、W&B UIで視覚化できます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合や、run がオフラインの場合。 |

### `get_added_local_path_name`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1585-L1597)

```python
get_added_local_path_name(
    local_path: str
) -> Optional[str]
```

ローカルファイルシステムのパスから追加されたファイルのアーティファクト相対名を取得します。

| 引数 |  |
| :--- | :--- |
|  `local_path` |  アーティファクト相対名に解決されるローカルパス。 |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクト相対名。 |

### `get_entry`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1515-L1535)

```python
get_entry(
    name: StrPath
) -> ArtifactManifestEntry
```

指定された名前でエントリを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクト相対名。 |

| 戻り値 |  |
| :--- | :--- |
|  `W&B` オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合や、run がオフラインの場合。 |
|  `KeyError` |  アーティファクトに指定された名前のエントリが含まれていない場合。 |

### `get_path`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1507-L1513)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

非推奨。`get_entry(name)`を使用してください。

### `is_draft`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L741-L746)

```python
is_draft() -> bool
```

アーティファクトが保存されていないかどうかを確認します。

戻り値: Boolean。アーティファクトが保存されている場合は `False`。保存されていない場合は `True`。

### `json_encode`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2229-L2236)

```python
json_encode() -> Dict[str, Any]
```

アーティファクトを JSON 形式でエンコードして返します。

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトの属性を表す `string` キーを持つ `dict`。 |

### `link`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2065-L2093)

```python
link(
    target_path: str,
    aliases: Optional[List[str]] = None
) -> None
```

このアーティファクトをポートフォリオ (アーティファクトの昇進されたコレクション) にリンクします。

| 引数 |  |
| :--- | :--- |
|  `target_path` |  プロジェクト内のポートフォリオへのパス。ターゲットパスは次のいずれかのスキーマに従う必要があります: `{portfolio}`, `{project}/{portfolio}` または `{entity}/{project}/{portfolio}`。アーティファクトをモデルレジストリにリンクする場合は、`{"model-registry"}/{Registered Model Name}` または `{entity}/{"model-registry"}/{Registered Model Name}` のスキーマを使用します。 |
|  `aliases` |  指定されたポートフォリオ内でアーティファクトを一意に識別する文字列のリスト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `logged_by`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2184-L2227)

```python
logged_by() -> Optional[Run]
```

このアーティファクトを最初にログに記録した W&B run を取得します。

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトを最初にログに記録した W&B run の名前。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `new_draft`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L345-L377)

```python
new_draft() -> "Artifact"
```

このコミットされたアーティファクトと同じ内容を持つ新しいドラフトのアーティファクトを作成します。

返されるアーティファクトは拡張または変更され、新しいバージョンとしてログに記録できます。

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `new_file`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1115-L1152)

```python
@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "w",
    encoding: Optional[str] = None
) -> Generator[IO, None, None]
```

新しい一時ファイルを開き、アーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  アーティファクトに追加する新しいファイルの名前。 |
|  `mode` |  新しいファイルを開くために使用するファイルアクセスモード。 |
|  `encoding` |  新しいファイルを開くために使用するエンコーディング。 |

| 戻り値 |  |
| :--- | :--- |
|  書き込み可能な新しいファイルオブジェクト。閉じると、自動的にアーティファクトに追加されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |

### `path_contains_dir_prefix`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1664-L1671)

```python
@classmethod
path_contains_dir_prefix(
    path: StrPath,
    dir_path: StrPath
) -> bool
```

`path` が `dir_path` をプレフィックスとして含むかどうかを返します。

### `remove`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1476-L1505)

```python
remove(
    item: Union[StrPath, 'ArtifactManifestEntry']
) -> None
```

アーティファクトからアイテムを削除します。

| 引数 |  |
| :--- | :--- |
|  `item` |  削除するアイテム。特定のマニフェストエントリまたはアーティファクト相対パスの名前。アイテムがディレクトリに一致する場合、そのディレクトリ内のすべてのアイテムが削除されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンに変更を加えることはできません。代わりに新しいアーティファクトバージョンをログに記録してください。 |
|  `FileNotFoundError` |  アーティファクトにアイテムが見つからない場合。 |

### `save`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L751-L790)

```python
save(
    project: Optional[str] = None,
    settings: Optional['wandb.wandb_sdk.wandb_settings.Settings'] = None
) -> None
```

アーティファクトに対して行った変更を保存します。

現在 run 内にいる場合、その run がこのアーティファクトをログに記録します。現在 run 内にいない場合、自動でこのアーティファクトを追跡する "auto" タイプの run が作成されます。

| 引数 |  |
| :--- | :--- |
|  `project` |  run がすでに存在する場合に使用するプロジェクト。 |
|  `settings` |  自動 run を初期化するときに使用する設定オブジェクト。主にテストハーネスで使用されます。 |

### `should_download_entry`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1673-L1679)

```python
@classmethod
should_download_entry(
    entry: ArtifactManifestEntry,
    prefix: Optional[StrPath]
) -> bool
```

### `unlink`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2095-L2111)

```python
unlink() -> None
```

アーティファクトがポートフォリオ（アーティファクトの昇進されたコレクション）のメンバーである場合、そのリンクを解除します。

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |
|  `ValueError` |  アーティファクトがリンクされていない場合、例えばポートフォリオコレクションのメンバーではない場合。 |

### `used_by`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L2137-L2182)

```python
used_by() -> List[Run]
```

このアーティファクトを使用した run のリストを取得します。

| 戻り値 |  |
| :--- | :--- |
|  `Run` オブジェクトのリスト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |

### `verify`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1909-L1948)

```python
verify(
    root: Optional[str] = None
) -> None
```

アーティファクトの内容がマニフェストと一致するかどうかを確認します。

ディレクトリ内のすべてのファイルのチェックサムが取られ、チェックサムがアーティファクトのマニフェストとクロスチェックされます。参照は検証されません。

| 引数 |  |
| :--- | :--- |
|  `root` |  検証するディレクトリ。None の場合、アーティファクトは './artifacts/self.name/' にダウンロードされます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合。 |
|  `ValueError` |  検証が失敗した場合。 |

### `wait`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L798-L819)

```python
wait(
    timeout: Optional[int] = None
) -> "Artifact"
```

必要に応じて、このアーティファクトのログ記録が完了するまで待機します。

| 引数 |  |
| :--- | :--- |
|  `timeout` |  待機する時間（秒）。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `__getitem__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1085-L1097)

```python
__getitem__(
    name: str
) -> Optional[data_types.WBValue]
```

アーティファクト相対 `name` に位置する WBValue オブジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクト相対名。 |

| 戻り値 |  |
| :--- | :--- |
|  `wandb.log()` を使用してログに記録され、W&B UI で視覚化できる W&B オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログに記録されていない場合、または run がオフラインの場合。 |

### `__setitem__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/artifacts/artifact.py#L1099-L1113)

```python
__setitem__(
    name: str,
    item: data_types.WBValue
) -> ArtifactManifestEntry
```

`item` をパス `name` にアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  オブジェクトを追加するアーティファクト内のパス。 |
|  `item` |  追加するオブジェクト。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

