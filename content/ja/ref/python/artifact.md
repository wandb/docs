---
title: アーティファクト
menu:
  reference:
    identifier: ja-ref-python-artifact
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L96-L2410 >}}

データセットとモデルのバージョン管理のための柔軟で軽量なビルディングブロック。

```python
Artifact(
    name: str,
    type: str,
    description: (str | None) = None,
    metadata: (dict[str, Any] | None) = None,
    incremental: bool = (False),
    use_as: (str | None) = None
) -> None
```

空の W&B Artifact を構築します。アーティファクトの内容を `add` で始まるメソッドを用いて追加してください。すべてのファイルがそろったら `wandb.log_artifact()` を呼び出してログ化します。

| 引数 |  |
| :--- | :--- |
|  `name` |  アーティファクトの人間が読める名前です。この名前を W&B App UI 内で特定のアーティファクトを識別するのに使用します。プログラムでアーティファクトを参照するには `use_artifact` パブリック API を使用します。名前には文字、数字、アンダースコア、ハイフン、ドットを含めることができます。名前はプロジェクト内で一意である必要があります。 |
|  `type` |  アーティファクトのタイプです。アーティファクトの分類や区別に使用します。文字、数字、アンダースコア、ハイフン、ドットを含む任意の文字列を使用できます。一般的なタイプには `dataset` や `model` があります。アーティファクトを W&B Model Registry にリンクしたい場合は、タイプ文字列に `model` を含めてください。 |
|  `description` |  アーティファクトの説明です。Model または Dataset Artifacts の場合、標準化されたチームのモデルカードまたはデータセットカードについてのドキュメントを追加します。アーティファクトの説明はプログラムで `Artifact.description` 属性を使って表示できます。 W&B App では説明がマークダウンとしてレンダリングされます。 |
|  `metadata` |  アーティファクトに関する追加情報を提供します。メタデータはキーと値のペアの辞書として指定します。合計で最大100個のキーを指定できます。 |
|  `incremental` |  既存のアーティファクトを修正するには代わりに `Artifact.new_draft()` メソッドを使用します。 |
|  `use_as` |  W&B Launch 専用のパラメータです。一般的な使用には推奨されません。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

| 属性 |  |
| :--- | :--- |
|  `aliases` |  アーティファクトのバージョンに割り当てられた意味的にフレンドリーな参照または識別用の「ニックネーム」のリストです。エイリアスはプログラムで参照できる可変な参照です。アーティファクトのエイリアスは W&B App UI またはプログラムで変更できます。詳しくは [Create new artifact versions](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version) を参照してください。 |
|  `collection` |  このアーティファクトが取得されたコレクションです。コレクションはアーティファクトのバージョンをまとめた順序付けられたグループです。このアーティファクトがポートフォリオ/リンクされたコレクションから取得された場合、そのコレクションが返されます。アーティファクトの起源となるコレクションはソースシーケンスとして知られています。 |
|  `commit_hash` |  このアーティファクトがコミットされた時に返されたハッシュです。 |
|  `created_at` |  アーティファクトが作成された時のタイムスタンプです。 |
|  `description` |  アーティファクトの説明です。 |
|  `digest` |  アーティファクトの論理的なダイジェストです。ダイジェストはアーティファクトの内容のチェックサムです。アーティファクトのダイジェストが現在の `latest` バージョンと同じであれば、`log_artifact` は何もせず終了します。 |
|  `entity` |  セカンダリ（ポートフォリオ）アーティファクトコレクションのエンティティの名前です。 |
|  `file_count` |  ファイルの数（参照を含む）です。 |
|  `id` |  アーティファクトの ID です。 |
|  `manifest` |  アーティファクトのマニフェストです。マニフェストには全ての内容が記載されており、アーティファクトがログ化された後は変更できません。 |
|  `metadata` |  ユーザー定義のアーティファクトメタデータです。アーティファクトに関連付けられた構造化データです。 |
|  `name` |  セカンダリ（ポートフォリオ）コレクションでのアーティファクト名とバージョンです。文字列のフォーマットは `{collection}:{alias}` です。アーティファクトが保存される前は、バージョンがまだ知られていないため名前のみを含みます。 |
|  `project` |  セカンダリ（ポートフォリオ）アーティファクトコレクションのプロジェクト名です。 |
|  `qualified_name` |  セカンダリ（ポートフォリオ）コレクションの entity/project/name です。 |
|  `size` |  アーティファクトの総サイズ（バイト単位）です。このアーティファクトに追跡されているすべての参照を含みます。 |
|  `source_collection` |  アーティファクトのプライマリ（シーケンス）コレクションです。 |
|  `source_entity` |  プライマリ（シーケンス）アーティファクトコレクションのエンティティの名前です。 |
|  `source_name` |  プライマリ（シーケンス）コレクションでのアーティファクト名とバージョンです。文字列のフォーマットは `{collection}:{alias}` です。アーティファクトが保存される前は、バージョンがまだ知られていないため名前のみを含みます。 |
|  `source_project` |  プライマリ（シーケンス）アーティファクトコレクションのプロジェクト名です。 |
|  `source_qualified_name` |  プライマリ（シーケンス）コレクションの entity/project/name です。 |
|  `source_version` |  プライマリ（シーケンス）コレクションでのアーティファクトのバージョンです。文字列のフォーマットは `v{number}` です。 |
|  `state` |  アーティファクトの状態です。 "PENDING", "COMMITTED", "DELETED" のいずれかです。 |
|  `tags` |  このアーティファクトバージョンに割り当てられたタグのリストです。 |
|  `ttl` |  アーティファクトの生存期間 (TTL) ポリシーです。TTL ポリシーの期間が経過すると、アーティファクトはすぐに削除されます。`None` に設定されている場合、アーティファクトは TTL ポリシーを無効にし、チームのデフォルト TTL が存在しても削除が予定されません。TTL は、チーム管理者がデフォルトの TTL を設定した場合と、アーティファクトにカスタムポリシーがない場合は、チームのデフォルト TTL からポリシーを継承します。 |
|  `type` |  アーティファクトのタイプです。一般的なタイプには `dataset` や `model` が含まれます。 |
|  `updated_at` |  アーティファクトが最後に更新された時刻です。 |
|  `url` |  アーティファクトの URL を構築します。 |
|  `version` |  セカンダリ（ポートフォリオ）コレクションでのアーティファクトのバージョンです。 |

## メソッド

### `add`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1487-L1578)

```python
add(
    obj: WBValue,
    name: StrPath,
    overwrite: bool = (False)
) -> ArtifactManifestEntry
```

wandb.WBValue `obj` をアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `obj` |  追加するオブジェクトです。現在、Bokeh、JoinedTable、PartitionedTable、Table、Classes、ImageMask、BoundingBoxes2D、Audio、Image、Video、Html、Object3D のいずれかをサポートしています。 |
|  `name` |  オブジェクトを追加するアーティファクト内のパスです。 |
|  `overwrite` |  True の場合、同じファイルパスを持つ既存のオブジェクトを上書きします（該当する場合）。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンには変更を加えることができません。新しいアーティファクトバージョンをログに記録してください。 |

### `add_dir`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1342-L1402)

```python
add_dir(
    local_path: str,
    name: (str | None) = None,
    skip_cache: (bool | None) = (False),
    policy: (Literal['mutable', 'immutable'] | None) = "mutable"
) -> None
```

ローカルディレクトリーをアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `local_path` |  ローカルディレクトリーのパスです。 |
|  `name` |  アーティファクト内のサブディレクトリ名です。指定した名前は W&B App UI にアーティファクトの `type` にネストされて表示されます。デフォルトではアーティファクトのルートになります。 |
|  `skip_cache` |  `True` に設定すると、W&B はアップロード時にファイルをキャッシュにコピー/移動しません。 |
|  `policy` |  "mutable" | "immutable"。デフォルトは "mutable" です。"mutable": アップロード中の破損を防ぐためにファイルの一時コピーを作成します。"immutable": 保護を無効にし、ファイルを削除したり変更したりしないようユーザーに依存します。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンには変更を加えることができません。新しいアーティファクトバージョンをログに記録してください。 |
|  `ValueError` |  ポリシーは "mutable" または "immutable" でなければなりません。 |

### `add_file`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1289-L1340)

```python
add_file(
    local_path: str,
    name: (str | None) = None,
    is_tmp: (bool | None) = (False),
    skip_cache: (bool | None) = (False),
    policy: (Literal['mutable', 'immutable'] | None) = "mutable",
    overwrite: bool = (False)
) -> ArtifactManifestEntry
```

ローカルファイルをアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `local_path` |  追加されるファイルのパスです。 |
|  `name` |  追加されるファイルに使用するアーティファクト内のパスです。デフォルトではファイルのベース名になります。 |
|  `is_tmp` |  True の場合、名前の競合を避けるためにファイルの名前が決定論的に変更されます。 |
|  `skip_cache` |  `True` の場合、アップロード後にファイルをキャッシュにコピーしません。 |
|  `policy` |  デフォルトは "mutable" に設定されています。"mutable" の場合、アップロード中の破損を防ぐためにファイルの一時コピーを作成します。"immutable" の場合、保護を無効にし、ファイルを削除や変更しないようユーザーに依存します。 |
|  `overwrite` |  `True` の場合、既に存在するファイルを上書きします。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンには変更を加えることができません。新しいアーティファクトバージョンをログに記録してください。 |
|  `ValueError` |  ポリシーは "mutable" または "immutable" でなければなりません。 |

### `add_reference`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1404-L1485)

```python
add_reference(
    uri: (ArtifactManifestEntry | str),
    name: (StrPath | None) = None,
    checksum: bool = (True),
    max_objects: (int | None) = None
) -> Sequence[ArtifactManifestEntry]
```

URI によって示される参照をアーティファクトに追加します。

アーティファクトにファイルやディレクトリを追加する場合と異なり、参照は W&B にアップロードされません。詳細は、[Track external files](https://docs.wandb.ai/guides/artifacts/track-external-files) を参照してください。

デフォルトでは、以下のスキームがサポートされています:

- http(s): ファイルのサイズとダイジェストはサーバーから返された `Content-Length` と `ETag` レスポンスヘッダによって推測されます。
- s3: チェックサムとサイズはオブジェクトメタデータから取得されます。バケットバージョン管理が有効な場合、バージョン ID も追跡されます。
- gs: チェックサムとサイズはオブジェクトメタデータから取得されます。バケットバージョン管理が有効な場合、バージョン ID も追跡されます。
- https, `*.blob.core.windows.net` (Azure) ドメイン: チェックサムとサイズはブロブメタデータから取得されます。ストレージアカウントのバージョン管理が有効な場合、バージョン ID も追跡されます。
- file: チェックサムとサイズはファイルシステムから取得されます。このスキームは、アップロードする必要はないが追跡したいファイルを含む NFS シェアや他の外部マウントボリュームを持っている場合に便利です。

その他のスキームについては、ダイジェストは URI のハッシュとして生成され、サイズは空欄のままです。

| 引数 |  |
| :--- | :--- |
|  `uri` |  追加する参照の URI パスです。URI パスは `Artifact.get_entry` から返されたオブジェクトであり、他のアーティファクトのエントリへの参照として保存することができます。 |
|  `name` |  この参照の内容を置くアーティファクト内のパスです。 |
|  `checksum` |  参照 URI にあるリソースをチェックサムするかどうか。チェックサムは自動的な整合性の検証を可能にするため、非常に推奨されます。チェックサムを無効にすると、アーティファクトの作成が速くなりますが、参照ディレクトリは繰り返し処理されないため、ディレクトリ内のオブジェクトはアーティファクトに保存されません。参照オブジェクトを追加する場合は `checksum=False` を設定することをお勧めします。そうすれば、参照 URI が変更された場合のみ新しいバージョンが作成されます。 |
|  `max_objects` |  ディレクトリまたはバケットストアプレフィックスを指す参照を追加する際に考慮する最大オブジェクト数です。デフォルトでは、Amazon S3、GCS、Azure、ローカルファイルに対して許可されている最大オブジェクト数は 10,000,000 です。他の URI スキーマには最大はありません。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンには変更を加えることができません。新しいアーティファクトバージョンをログに記録してください。 |

### `checkout`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1993-L2021)

```python
checkout(
    root: (str | None) = None
) -> str
```

指定されたルートディレクトリをアーティファクトの内容に置き換えます。

警告: `root` に含まれているがアーティファクトに含まれていないすべてのファイルは削除されます。

| 引数 |  |
| :--- | :--- |
|  `root` |  このアーティファクトのファイルで置き換えるディレクトリ。 |

| 戻り値 |  |
| :--- | :--- |
|  チェックアウトされた内容のパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |

### `delete`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2131-L2150)

```python
delete(
    delete_aliases: bool = (False)
) -> None
```

アーティファクトおよびそのファイルを削除します。

リンクされたアーティファクト（すなわち、ポートフォリオコレクションのメンバー）に対して呼び出された場合、リンクのみが削除され、元のアーティファクトには影響を与えません。

| 引数 |  |
| :--- | :--- |
|  `delete_aliases` |  `True` に設定されている場合は、アーティファクトに関連付けられたすべてのエイリアスを削除します。それ以外の場合、既存のエイリアスがある場合は例外を発生させます。このパラメータは、アーティファクトがリンクされている場合（つまり、ポートフォリオコレクションのメンバー）には無視されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |

### `download`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1756-L1807)

```python
download(
    root: (StrPath | None) = None,
    allow_missing_references: bool = (False),
    skip_cache: (bool | None) = None,
    path_prefix: (StrPath | None) = None
) -> FilePathStr
```

アーティファクトの内容を指定されたルートディレクトリにダウンロードします。

`root` 内の既存ファイルは変更されません。 `root` の内容をアーティファクトと完全に一致させたい場合は、`download` を呼び出す前に `root` を明示的に削除してください。

| 引数 |  |
| :--- | :--- |
|  `root` |  W&B がアーティファクトのファイルを保存するディレクトリ。 |
|  `allow_missing_references` |  `True` に設定した場合、ダウンロード中に無効な参照パスが無視されます。 |
|  `skip_cache` |  `True` に設定した場合、ダウンロード中にアーティファクトキャッシュはスキップされ、W&B は各ファイルをデフォルトのルートまたは指定されたダウンロードディレクトリにダウンロードします。 |
|  `path_prefix` |  指定すると、そのプレフィックスで始まるパスを持つファイルのみがダウンロードされます。Unix 形式（フォワードスラッシュ）を使用します。 |

| 戻り値 |  |
| :--- | :--- |
|  ダウンロードされた内容のパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |
|  `RuntimeError` |  オフラインモードでアーティファクトをダウンロードしようとした場合。 |

### `file`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2063-L2087)

```python
file(
    root: (str | None) = None
) -> StrPath
```

指定した `root` のディレクトリに単一のファイルアーティファクトをダウンロードします。

| 引数 |  |
| :--- | :--- |
|  `root` |  ファイルを保存するルートディレクトリです。デフォルトは `'./artifacts/self.name/'` です。 |

| 戻り値 |  |
| :--- | :--- |
|  ダウンロードされたファイルのフルパス。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |
|  `ValueError` |  アーティファクトに複数のファイルが含まれている場合。 |

### `files`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2089-L2106)

```python
files(
    names: (list[str] | None) = None,
    per_page: int = 50
) -> ArtifactFiles
```

このアーティファクトに保存されているすべてのファイルを反復処理します。

| 引数 |  |
| :--- | :--- |
|  `names` |  アーティファクトのルートからの相対ファイルパスで、リストを希望するファイルのパス。 |
|  `per_page` |  要求ごとに返されるファイルの数。 |

| 戻り値 |  |
| :--- | :--- |
|  `File` オブジェクトを含むイテレータ。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |

### `finalize`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L902-L910)

```python
finalize() -> None
```

アーティファクトバージョンを確定します。

一度確定されたアーティファクトバージョンは、特定のアーティファクトバージョンとしてログされるため、変更できません。新しいデータをアーティファクトに記録するには、新しいアーティファクトバージョンを作成してください。アーティファクトは、`log_artifact` を使用してログ化すると自動的に確定されます。

### `get`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1673-L1718)

```python
get(
    name: str
) -> (WBValue | None)
```

アーティファクト相対 `name` に配置されている WBValue オブジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクトの相対名。 |

| 戻り値 |  |
| :--- | :--- |
|  `wandb.log()` で記録され、W&B UI で視覚化可能な W&B オブジェクトです。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合や、run がオフラインの場合。 |

### `get_added_local_path_name`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1720-L1732)

```python
get_added_local_path_name(
    local_path: str
) -> (str | None)
```

ローカルファイルシステムパスによって追加されたファイルのアーティファクト相対名を取得します。

| 引数 |  |
| :--- | :--- |
|  `local_path` |  アーティファクト相対名に解決するローカルパス。 |

| 戻り値 |  |
| :--- | :--- |
|  アーティファクト相対名。 |

### `get_entry`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1652-L1671)

```python
get_entry(
    name: StrPath
) -> ArtifactManifestEntry
```

指定した名前のエントリを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクト相対名です。 |

| 戻り値 |  |
| :--- | :--- |
|  `W&B` オブジェクトです。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合、または run がオフラインの場合。 |
|  `KeyError` |  指定した名前のエントリがアーティファクトに含まれていない場合。 |

### `get_path`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1644-L1650)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

非推奨。`get_entry(name)` を使用してください。

### `is_draft`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L912-L917)

```python
is_draft() -> bool
```

アーティファクトが保存されていないかをチェックします。

Returns: Boolean. `False` はアーティファクトが保存された場合。`True` はアーティファクトが保存されていない場合。

### `json_encode`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2338-L2345)

```python
json_encode() -> dict[str, Any]
```

アーティファクトを JSON 形式にエンコードして返します。

| 戻り値 |  |
| :--- | :--- |
|  アーティファクトの属性を表す `string` キーを持つ `dict` 。 |

### `link`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2177-L2205)

```python
link(
    target_path: str,
    aliases: (list[str] | None) = None
) -> None
```

このアーティファクトをポートフォリオ（プロモートされたアーティファクトのコレクション）にリンクします。

| 引数 |  |
| :--- | :--- |
|  `target_path` |  プロジェクト内のポートフォリオへのパス。ターゲットパスは、次のスキーマのいずれかに従っている必要があります。`{portfolio}`, `{project}/{portfolio}` または `{entity}/{project}/{portfolio}`。Model Registry にアーティファクトをリンクする場合は、プロジェクト内のジェネリックポートフォリオではなく、ターゲットパスを次のスキーマ `{model-registry}/{Registered Model Name}` または `{entity}/{model-registry}/{Registered Model Name}` に設定します。 |
|  `aliases` |  指定されたポートフォリオ内でアーティファクトを一意に識別する文字列のリスト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |

### `logged_by`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2294-L2336)

```python
logged_by() -> (Run | None)
```

元々アーティファクトをログした W&B run を取得します。

| 戻り値 |  |
| :--- | :--- |
|  元々アーティファクトをログした W&B run の名前。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |

### `new_draft`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L424-L457)

```python
new_draft() -> Artifact
```

このコミット済みのアーティファクトと同じ内容の新しいドラフトアーティファクトを作成します。

既存のアーティファクトを修正すると、新しいアーティファクトバージョンである「インクリメンタルアーティファクト」が作成されます。返されたアーティファクトは拡張または修正され、新しいバージョンとしてログに記録できます。

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |

### `new_file`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1246-L1287)

```python
@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "x",
    encoding: (str | None) = None
) -> Iterator[IO]
```

新しい一時ファイルを開いてアーティファクトに追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  アーティファクトに追加する新しいファイルの名前。 |
|  `mode` |  新しいファイルを開くために使用するファイルアクセスモード。 |
|  `encoding` |  新しいファイルを開く際に使用するエンコーディング。 |

| 戻り値 |  |
| :--- | :--- |
|  書き込み可能な新しいファイルオブジェクト。閉じると、ファイルは自動的にアーティファクトに追加されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンには変更を加えることができません。新しいアーティファクトバージョンをログに記録してください。 |

### `remove`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1614-L1642)

```python
remove(
    item: (StrPath | ArtifactManifestEntry)
) -> None
```

アイテムをアーティファクトから削除します。

| 引数 |  |
| :--- | :--- |
|  `item` |  削除するアイテム。特定のマニフェストエントリまたはアーティファクト相対パスの名前であることができます。アイテムがディレクトリに一致する場合、そのディレクトリ内のすべてのアイテムが削除されます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンには変更を加えることができません。新しいアーティファクトバージョンをログに記録してください。 |
|  `FileNotFoundError` |  アイテムがアーティファクト内で見つからない場合。 |

### `save`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L922-L961)

```python
save(
    project: (str | None) = None,
    settings: (wandb.Settings | None) = None
) -> None
```

アーティファクトに加えた変更を永続化します。

現在 run にいる場合、その run はこのアーティファクトをログに記録します。run にいない場合、アーティファクトを追跡するために "auto" タイプの run が作成されます。

| 引数 |  |
| :--- | :--- |
|  `project` |  run がすでにコンテキスト内にない場合にアーティファクトに使用するプロジェクト。 |
|  `settings` |  自動 run を初期化する際に使用する設定オブジェクト。主にテストハーネスに使用されます。 |

### `unlink`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2207-L2222)

```python
unlink() -> None
```

このアーティファクトが現在ポートフォリオのメンバーである場合、それを解除します（プロモートされたアーティファクトのコレクション）。

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |
|  `ValueError` |  アーティファクトがリンクされていない場合、つまりポートフォリオコレクションのメンバーでない場合。 |

### `used_by`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2248-L2292)

```python
used_by() -> list[Run]
```

このアーティファクトを使用した run のリストを取得します。

| 戻り値 |  |
| :--- | :--- |
|  `Run` オブジェクトのリスト。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |

### `verify`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2023-L2061)

```python
verify(
    root: (str | None) = None
) -> None
```

アーティファクトの内容がマニフェストと一致するかを確認します。

ディレクトリ内のすべてのファイルはチェックサムが計算され、チェックサムはアーティファクトのマニフェストと照合されます。参照は確認されません。

| 引数 |  |
| :--- | :--- |
|  `root` |  検証するディレクトリ。None の場合、アーティファクトは `'./artifacts/self.name/'` にダウンロードされます。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合。 |
|  `ValueError` |  検証に失敗した場合。 |

### `wait`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L971-L995)

```python
wait(
    timeout: (int | None) = None
) -> Artifact
```

必要であれば、このアーティファクトがログの終了を待ちます。

| 引数 |  |
| :--- | :--- |
|  `timeout` |  待機する時間（秒単位）。 |

| 戻り値 |  |
| :--- | :--- |
|  `Artifact` オブジェクト。 |

### `__getitem__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1216-L1228)

```python
__getitem__(
    name: str
) -> (WBValue | None)
```

アーティファクト相対 `name` に配置されている WBValue オブジェクトを取得します。

| 引数 |  |
| :--- | :--- |
|  `name` |  取得するアーティファクトの相対名。 |

| 戻り値 |  |
| :--- | :--- |
|  `wandb.log()` で記録され、W&B UI で視覚化可能な W&B オブジェクトです。 |

| 例外 |  |
| :--- | :--- |
|  `ArtifactNotLoggedError` |  アーティファクトがログされていない場合や、run がオフラインの場合。 |

### `__setitem__`

[ソースを表示](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1230-L1244)

```python
__setitem__(
    name: str,
    item: WBValue
) -> ArtifactManifestEntry
```

`item` をアーティファクトの `name` パスに追加します。

| 引数 |  |
| :--- | :--- |
|  `name` |  オブジェクトを追加するアーティファクト内のパス。 |
|  `item` |  追加するオブジェクト。 |

| 戻り値 |  |
| :--- | :--- |
|  追加されたマニフェストエントリ |

| 例外 |  |
| :--- | :--- |
|  `ArtifactFinalizedError` |  現在のアーティファクトバージョンには変更を加えることができません。新しいアーティファクトバージョンをログに記録してください。 |