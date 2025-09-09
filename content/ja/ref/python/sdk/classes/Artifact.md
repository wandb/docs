---
title: アーティファクト
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-sdk-classes-Artifact
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/artifacts/artifact.py >}}




## <kbd>class</kbd> `Artifact`
データセットやモデルのバージョン管理のための、柔軟で軽量なビルディングブロック。

空の W&B `Artifact` を作成します。`add` で始まるメソッドでアーティファクトの中身を追加してください。必要なファイルをすべて含めたら、`run.log_artifact()` を呼び出してログできます。



**Args:**
 
 - `name` (str):  アーティファクトの読みやすい名前。W&B App UI またはプログラムから、特定のアーティファクトを識別するために使用します。`use_artifact` Public API を使って対話的にアーティファクトを参照できます。名前には英数字、アンダースコア、ハイフン、ドットが使用できます。名前はプロジェクト内で一意である必要があります。 
 - `type` (str):  アーティファクトのタイプ。アーティファクトのタイプを使って整理・区別します。英数字、アンダースコア、ハイフン、ドットを含む任意の文字列が使えます。一般的なタイプは `dataset` や `model` です。W&B モデルレジストリにアーティファクトをリンクしたい場合は、タイプ文字列に `model` を含めてください。内部用途のために予約されており、ユーザーが設定できないタイプもあります。例えば `job` や、`wandb-` で始まるタイプです。 
 - `description (str | None) = None`:  アーティファクトの説明。Model や Dataset のアーティファクトの場合は、チームで標準化したモデルカードやデータセットカードのドキュメントを追加します。アーティファクトの説明は、`Artifact.description` 属性からプログラムで、または W&B App UI で閲覧できます。説明は W&B App で Markdown としてレンダリングされます。 
 - `metadata (dict[str, Any] | None) = None`:  アーティファクトに関する追加情報。キーと値のペアからなる辞書で指定します。キーは合計 100 個までです。 
 - `incremental`:  既存のアーティファクトを変更するには、代わりに `Artifact.new_draft()` メソッドを使用してください。 
 - `use_as`:  廃止予定。 
 - `is_link`:  このアーティファクトがリンクされたアーティファクト（`True`）か、ソースアーティファクト（`False`）かのブール値。 



**Returns:**
 `Artifact` オブジェクト。 

### <kbd>method</kbd> `Artifact.__init__`

```python
__init__(
    name: 'str',
    type: 'str',
    description: 'str | None' = None,
    metadata: 'dict[str, Any] | None' = None,
    incremental: 'bool' = False,
    use_as: 'str | None' = None
) → None
```






---

### <kbd>property</kbd> Artifact.aliases

1 つ以上の意味のある参照、または 

アーティファクトのバージョンに付けられた識別用の「ニックネーム」。 

エイリアスはプログラムから参照できる可変参照です。W&B App UI またはプログラムからアーティファクトのエイリアスを変更できます。詳しくは [Create new artifact versions](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version) を参照してください。 

---

### <kbd>property</kbd> Artifact.collection

このアーティファクトが取得されたコレクション。 

コレクションはアーティファクトのバージョンの順序付きグループです。このアーティファクトがポートフォリオ / リンクされたコレクションから取得された場合、元のアーティファクトバージョンのコレクションではなく、そのコレクションが返されます。アーティファクトが元々属していたコレクションはソースシーケンスと呼ばれます。 

---

### <kbd>property</kbd> Artifact.commit_hash

このアーティファクトをコミットしたときに返されたハッシュ。 

---

### <kbd>property</kbd> Artifact.created_at

アーティファクトが作成された時刻のタイムスタンプ。 

---

### <kbd>property</kbd> Artifact.description

アーティファクトの説明。 

---

### <kbd>property</kbd> Artifact.digest

アーティファクトの論理ダイジェスト。 

ダイジェストはアーティファクト内容のチェックサムです。アーティファクトのダイジェストが現在の `latest` バージョンと同じ場合、`log_artifact` は no-op（何もしません）になります。 

---


### <kbd>property</kbd> Artifact.entity

アーティファクトのコレクションが属するエンティティ名。 

アーティファクトがリンクの場合、リンク先アーティファクトのエンティティになります。 

---

### <kbd>property</kbd> Artifact.file_count

ファイル数（参照を含む）。 

---

### <kbd>property</kbd> Artifact.history_step

このアーティファクトのソース run が履歴メトリクスをログした最も近い step。 



**Examples:**
 ```python
run = artifact.logged_by()
if run and (artifact.history_step is not None):
     history = run.sample_history(
         min_step=artifact.history_step,
         max_step=artifact.history_step + 1,
         keys=["my_metric"],
     )
``` 

---

### <kbd>property</kbd> Artifact.id

アーティファクトの ID。 

---


### <kbd>property</kbd> Artifact.is_link

このアーティファクトがリンクアーティファクトかどうかを示すブール値。 

True: このアーティファクトはソースアーティファクトへのリンク。False: このアーティファクトはソースアーティファクト。 

---

### <kbd>property</kbd> Artifact.linked_artifacts

ソースアーティファクトにリンクされているすべてのアーティファクトのリストを返します。 

このアーティファクトがリンクアーティファクト（`artifact.is_link == True`）の場合、空のリストを返します。最大 500 件まで。 

---

### <kbd>property</kbd> Artifact.manifest

アーティファクトのマニフェスト。 

マニフェストにはすべての内容が一覧され、アーティファクトをログした後は変更できません。 

---

### <kbd>property</kbd> Artifact.metadata

ユーザー定義のアーティファクトメタデータ。 

アーティファクトに関連付けられた構造化データ。 

---

### <kbd>property</kbd> Artifact.name

アーティファクトの名前とバージョン。 

`{collection}:{alias}` の形式の文字列。アーティファクトがログ / 保存される前に取得した場合、名前にはエイリアスは含まれません。アーティファクトがリンクの場合、名前はリンク先アーティファクトの名前になります。 

---

### <kbd>property</kbd> Artifact.project

アーティファクトのコレクションが属するプロジェクト名。 

アーティファクトがリンクの場合、リンク先アーティファクトのプロジェクトになります。 

---

### <kbd>property</kbd> Artifact.qualified_name

アーティファクトの entity/project/name。 

アーティファクトがリンクの場合、リンク先アーティファクトパスの qualified name になります。 

---

### <kbd>property</kbd> Artifact.size

アーティファクトの合計サイズ（バイト単位）。 

このアーティファクトが追跡している参照も含まれます。 

---

### <kbd>property</kbd> Artifact.source_artifact

ソースアーティファクトを返します。ソースアーティファクトとは、元々ログされたアーティファクトです。 

このアーティファクト自体がソースアーティファクト（`artifact.is_link == False`）の場合、自分自身を返します。 

---

### <kbd>property</kbd> Artifact.source_collection

アーティファクトのソースコレクション。 

ソースコレクションは、このアーティファクトがログされたコレクションです。 

---

### <kbd>property</kbd> Artifact.source_entity

ソースアーティファクトのエンティティ名。 

---

### <kbd>property</kbd> Artifact.source_name

ソースアーティファクトの名前とバージョン。 

`{source_collection}:{alias}` の形式の文字列。アーティファクトが保存される前は、バージョンが未定のため名前のみを含みます。 

---

### <kbd>property</kbd> Artifact.source_project

ソースアーティファクトのプロジェクト名。 

---

### <kbd>property</kbd> Artifact.source_qualified_name

ソースアーティファクトの source_entity/source_project/source_name。 

---

### <kbd>property</kbd> Artifact.source_version

ソースアーティファクトのバージョン。 

`v{number}` の形式の文字列。 

---

### <kbd>property</kbd> Artifact.state

アーティファクトのステータス。"PENDING"、"COMMITTED"、"DELETED" のいずれか。 

---

### <kbd>property</kbd> Artifact.tags

このアーティファクトのバージョンに付与された 1 つ以上のタグのリスト。 

---

### <kbd>property</kbd> Artifact.ttl

アーティファクトの TTL（time-to-live）ポリシー。 

TTL ポリシーの期間を過ぎると、アーティファクトは間もなく削除されます。`None` が設定されている場合、アーティファクトは TTL ポリシーを無効化し、チームのデフォルト TTL があっても削除対象にスケジュールされません。チーム管理者がデフォルト TTL を定義しており、アーティファクトにカスタムポリシーが設定されていない場合、アーティファクトはチームのデフォルト TTL ポリシーを継承します。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログまたは保存されていない場合、継承された TTL を取得できません。 

---

### <kbd>property</kbd> Artifact.type

アーティファクトのタイプ。一般的なタイプは `dataset` や `model`。 

---

### <kbd>property</kbd> Artifact.updated_at

アーティファクトが最後に更新された時刻。 

---

### <kbd>property</kbd> Artifact.url

アーティファクトの URL を組み立てます。 



**Returns:**
 
 - `str`:  アーティファクトの URL。 

---

### <kbd>property</kbd> Artifact.use_as

廃止予定。 

---

### <kbd>property</kbd> Artifact.version

アーティファクトのバージョン。 

`v{number}` の形式の文字列。アーティファクトがリンクアーティファクトの場合、バージョンはリンク先コレクションのものになります。 



---

### <kbd>method</kbd> `Artifact.add`

```python
add(
    obj: 'WBValue',
    name: 'StrPath',
    overwrite: 'bool' = False
) → ArtifactManifestEntry
```

`wandb.WBValue` の `obj` をこのアーティファクトに追加します。 



**Args:**
 
 - `obj`:  追加するオブジェクト。現在サポートしているのは、Bokeh、JoinedTable、PartitionedTable、Table、Classes、ImageMask、BoundingBoxes2D、Audio、Image、Video、Html、Object3D のいずれかです。 
 - `name`:  このオブジェクトをアーティファクト内のどのパスに追加するか。 
 - `overwrite`:  True の場合、同じファイルパスが存在すれば（該当する場合）上書きします。 



**Returns:**
 追加されたマニフェストエントリ。 



**Raises:**
 
 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため、変更できません。代わりに新しいアーティファクトバージョンをログしてください。 

---

### <kbd>method</kbd> `Artifact.add_dir`

```python
add_dir(
    local_path: 'str',
    name: 'str | None' = None,
    skip_cache: 'bool | None' = False,
    policy: "Literal['mutable', 'immutable'] | None" = 'mutable',
    merge: 'bool' = False
) → None
```

ローカルのディレクトリーをアーティファクトに追加します。 



**Args:**
 
 - `local_path`:  ローカルディレクトリーのパス。 
 - `name`:  アーティファクト内のサブディレクトリー名。指定した名前は、アーティファクトの `type` ごとにネストされて W&B App UI に表示されます。デフォルトはアーティファクトのルートです。 
 - `skip_cache`:  `True` の場合、アップロード中に W&B はファイルをキャッシュにコピー / 移動しません。 
 - `policy`:  既定は "mutable"。 
    - mutable: アップロード中の破損を防ぐため、一時コピーを作成します。 
    - immutable: 保護を無効にし、ユーザーがファイルを削除・変更しないことに依存します。 
 - `merge`:  `False`（デフォルト）の場合、以前の add_dir 呼び出しで既に追加され、その内容が変更されたファイルがあると ValueError を投げます。`True` の場合、内容が変更された既存ファイルを上書きします。新しいファイルは常に追加され、ファイルが削除されることはありません。ディレクトリー全体を置き換えるには、`add_dir(local_path, name=my_prefix)` のように名前を付けてディレクトリーを追加し、`remove(my_prefix)` を呼んでディレクトリーを削除してから、再度追加してください。 



**Raises:**
 
 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため、変更できません。代わりに新しいアーティファクトバージョンをログしてください。 
 - `ValueError`:  Policy は "mutable" または "immutable" でなければなりません。 

---

### <kbd>method</kbd> `Artifact.add_file`

```python
add_file(
    local_path: 'str',
    name: 'str | None' = None,
    is_tmp: 'bool | None' = False,
    skip_cache: 'bool | None' = False,
    policy: "Literal['mutable', 'immutable'] | None" = 'mutable',
    overwrite: 'bool' = False
) → ArtifactManifestEntry
```

ローカルファイルをアーティファクトに追加します。 



**Args:**
 
 - `local_path`:  追加するファイルのパス。 
 - `name`:  追加するファイルに対して、アーティファクト内で使用するパス。デフォルトはファイルのベース名。 
 - `is_tmp`:  True の場合、衝突を避けるためにファイル名を決定的にリネームします。 
 - `skip_cache`:  `True` の場合、アップロード後にファイルをキャッシュへコピーしません。 
 - `policy`:  既定は "mutable"。"mutable" の場合、アップロード中の破損を防ぐためにファイルの一時コピーを作成します。"immutable" の場合、保護を無効化し、ユーザーがファイルを削除・変更しないことに依存します。 
 - `overwrite`:  `True` の場合、ファイルが既に存在すれば上書きします。 



**Returns:**
 追加されたマニフェストエントリ。 



**Raises:**
 
 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため、変更できません。代わりに新しいアーティファクトバージョンをログしてください。 
 - `ValueError`:  Policy は "mutable" または "immutable" でなければなりません。 

---

### <kbd>method</kbd> `Artifact.add_reference`

```python
add_reference(
    uri: 'ArtifactManifestEntry | str',
    name: 'StrPath | None' = None,
    checksum: 'bool' = True,
    max_objects: 'int | None' = None
) → Sequence[ArtifactManifestEntry]
```

URI で示される参照をアーティファクトに追加します。 

アーティファクトに追加するファイルやディレクトリーとは異なり、参照は W&B にアップロードされません。詳しくは [Track external files](https://docs.wandb.ai/guides/artifacts/track-external-files) を参照してください。 

デフォルトで、以下のスキームがサポートされています: 


- http(s): サーバーが返す `Content-Length` と `ETag` レスポンスヘッダーにより、ファイルのサイズとダイジェストを推測します。 
- s3: チェックサムとサイズはオブジェクトのメタデータから取得します。バケットのバージョン管理が有効な場合は、バージョン ID も追跡します。 
- gs: チェックサムとサイズはオブジェクトのメタデータから取得します。バケットのバージョン管理が有効な場合は、バージョン ID も追跡します。 
- https、ドメインが `*.blob.core.windows.net` に一致 
- Azure: チェックサムとサイズは BLOB のメタデータから取得します。ストレージアカウントのバージョン管理が有効な場合は、バージョン ID も追跡します。 
- file: チェックサムとサイズはファイルシステムから取得します。NFS 共有や、アップロードはしないが追跡したいファイルを含む外部マウントボリュームがある場合に便利です。 

その他のスキームでは、ダイジェストは URI のハッシュのみとなり、サイズは空のままです。 



**Args:**
 
 - `uri`:  追加する参照の URI パス。URI パスは、`Artifact.get_entry` が返すオブジェクト（他のアーティファクトのエントリへの参照を保存するため）でも構いません。 
 - `name`:  この参照の内容をアーティファクト内のどのパスに配置するか。 
 - `checksum`:  参照 URI で示されるリソースに対してチェックサムを取るかどうか。整合性の自動検証が可能になるため、チェックサムは強く推奨します。チェックサムを無効にするとアーティファクトの作成は高速になりますが、参照ディレクトリーを走査しないため、そのディレクトリー内のオブジェクトはアーティファクトに保存されません。参照オブジェクトを追加する場合は `checksum=False` を設定することを推奨します。この場合、参照 URI が変わったときにだけ新しいバージョンが作成されます。 
 - `max_objects`:  ディレクトリーやバケットストアのプレフィックスを指す参照を追加する際に考慮するオブジェクトの最大数。デフォルトでは、Amazon S3、GCS、Azure、ローカルファイルについては最大 10,000,000 です。他の URI スキーマには上限はありません。 



**Returns:**
 追加されたマニフェストエントリ群。 



**Raises:**
 
 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため、変更できません。代わりに新しいアーティファクトバージョンをログしてください。 

---

### <kbd>method</kbd> `Artifact.checkout`

```python
checkout(root: 'str | None' = None) → str
```

指定したルートディレクトリーを、このアーティファクトの内容で置き換えます。 

警告: アーティファクトに含まれていない `root` 内のすべてのファイルは削除されます。 



**Args:**
 
 - `root`:  このアーティファクトのファイルで置き換えるディレクトリー。 



**Returns:**
 チェックアウトされた内容のパス。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 

---

### <kbd>method</kbd> `Artifact.delete`

```python
delete(delete_aliases: 'bool' = False) → None
```

アーティファクトとそのファイルを削除します。 

リンクされたアーティファクトで呼び出した場合、削除されるのはリンクのみで、ソースアーティファクトには影響しません。 

ソースアーティファクトとリンクアーティファクトの間のリンクを削除するには、`artifact.delete()` ではなく `artifact.unlink()` を使用してください。 



**Args:**
 
 - `delete_aliases`:  `True` の場合、アーティファクトに関連付けられたすべてのエイリアスを削除します。そうでない場合、エイリアスが存在すると例外を投げます。アーティファクトがリンクされている（ポートフォリオコレクションのメンバーである）場合、このパラメータは無視されます。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 

---

### <kbd>method</kbd> `Artifact.download`

```python
download(
    root: 'StrPath | None' = None,
    allow_missing_references: 'bool' = False,
    skip_cache: 'bool | None' = None,
    path_prefix: 'StrPath | None' = None,
    multipart: 'bool | None' = None
) → FilePathStr
```

アーティファクトの内容を、指定したルートディレクトリーへダウンロードします。 

`root` に既存のファイルがあっても変更されません。`root` の内容をアーティファクトと完全に一致させたい場合は、`download` を呼ぶ前に `root` を明示的に削除してください。 



**Args:**
 
 - `root`:  W&B がアーティファクトのファイルを保存するディレクトリー。 
 - `allow_missing_references`:  `True` の場合、参照ファイルのダウンロード時に無効な参照パスを無視します。 
 - `skip_cache`:  `True` の場合、ダウンロード時にアーティファクトキャッシュをスキップし、W&B は各ファイルをデフォルトのルートまたは指定したダウンロードディレクトリーに直接ダウンロードします。 
 - `path_prefix`:  指定すると、その接頭辞で始まるパスのファイルのみをダウンロードします。Unix 形式（スラッシュ区切り）を使用します。 
 - `multipart`:  `None`（デフォルト）の場合、個々のファイルサイズが 2GB を超えるときにマルチパートダウンロードで並列ダウンロードします。`True` または `False` に設定すると、ファイルサイズに関わらず、それぞれ並列または直列でダウンロードします。 



**Returns:**
 ダウンロードした内容のパス。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 

---

### <kbd>method</kbd> `Artifact.file`

```python
file(root: 'str | None' = None) → StrPath
```

単一ファイルのアーティファクトを、`root` で指定したディレクトリーにダウンロードします。 



**Args:**
 
 - `root`:  ファイルを保存するルートディレクトリー。デフォルトは `./artifacts/self.name/`。 



**Returns:**
 ダウンロードされたファイルのフルパス。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 
 - `ValueError`:  アーティファクトに 2 個以上のファイルが含まれている場合。 

---

### <kbd>method</kbd> `Artifact.files`

```python
files(names: 'list[str] | None' = None, per_page: 'int' = 50) → ArtifactFiles
```

このアーティファクトに保存されているすべてのファイルを反復処理します。 



**Args:**
 
 - `names`:  一覧表示したい、アーティファクトのルートからの相対ファイルパス。 
 - `per_page`:  1 回のリクエストで返すファイル数。 



**Returns:**
 `File` オブジェクトを含むイテレータ。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 

---

### <kbd>method</kbd> `Artifact.finalize`

```python
finalize() → None
```

アーティファクトバージョンを確定します。 

アーティファクトは特定のアーティファクトバージョンとしてログされるため、いったん確定するとそのバージョンは変更できません。アーティファクトにさらにデータをログするには、新しいアーティファクトバージョンを作成してください。`log_artifact` でアーティファクトをログすると、自動的に確定されます。 

---

### <kbd>method</kbd> `Artifact.get`

```python
get(name: 'str') → WBValue | None
```

アーティファクト相対 `name` の位置にある WBValue オブジェクトを取得します。 



**Args:**
 
 - `name`:  取得するアーティファクト相対名。 



**Returns:**
 `run.log()` でログでき、W&B App UI で可視化できる W&B オブジェクト。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない、または run がオフラインの場合。 

---

### <kbd>method</kbd> `Artifact.get_added_local_path_name`

```python
get_added_local_path_name(local_path: 'str') → str | None
```

ローカルファイルシステムパスから追加されたファイルの、アーティファクト相対名を取得します。 



**Args:**
 
 - `local_path`:  アーティファクト相対名に解決するローカルパス。 



**Returns:**
 アーティファクト相対名。 

---

### <kbd>method</kbd> `Artifact.get_entry`

```python
get_entry(name: 'StrPath') → ArtifactManifestEntry
```

指定した名前のエントリを取得します。 



**Args:**
 
 - `name`:  取得するアーティファクト相対名。 



**Returns:**
 `W&B` オブジェクト。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない、または run がオフラインの場合。 
 - `KeyError`:  指定した名前のエントリがアーティファクトに存在しない場合。 

---

### <kbd>method</kbd> `Artifact.get_path`

```python
get_path(name: 'StrPath') → ArtifactManifestEntry
```

非推奨。`get_entry(name)` を使用してください。 

---

### <kbd>method</kbd> `Artifact.is_draft`

```python
is_draft() → bool
```

アーティファクトが未保存かどうかを確認します。 



**Returns:**
  ブール値。アーティファクトが保存済みなら `False`、未保存なら `True`。 

---

### <kbd>method</kbd> `Artifact.json_encode`

```python
json_encode() → dict[str, Any]
```

アーティファクトを JSON 形式にエンコードして返します。 



**Returns:**
  アーティファクトの属性を表す、キーが `string` の `dict`。 

---

### <kbd>method</kbd> `Artifact.link`

```python
link(target_path: 'str', aliases: 'list[str] | None' = None) → Artifact
```

このアーティファクトをポートフォリオ（昇格されたアーティファクトのコレクション）にリンクします。 



**Args:**
 
 - `target_path`:  プロジェクト内のポートフォリオへのパス。ターゲットパスは `{portfolio}`、`{project}/{portfolio}`、`{entity}/{project}/{portfolio}` のいずれかのスキーマに従う必要があります。アーティファクトをプロジェクト内の一般的なポートフォリオではなく、モデルレジストリにリンクするには、`{"model-registry"}/{Registered Model Name}` または `{entity}/{"model-registry"}/{Registered Model Name}` のスキーマを `target_path` に設定してください。 
 - `aliases`:  指定したポートフォリオ内でアーティファクトを一意に識別する文字列リスト。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 



**Returns:**
 リンクされたアーティファクト。 

---

### <kbd>method</kbd> `Artifact.logged_by`

```python
logged_by() → Run | None
```

このアーティファクトを最初にログした W&B の run を取得します。 



**Returns:**
  このアーティファクトを最初にログした W&B の run の名前。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 

---

### <kbd>method</kbd> `Artifact.new_draft`

```python
new_draft() → Artifact
```

このコミット済みアーティファクトと同じ内容を持つ、新しいドラフトアーティファクトを作成します。 

既存のアーティファクトを変更すると、「インクリメンタルアーティファクト」として新しいアーティファクトバージョンが作成されます。返されるアーティファクトは拡張や変更が可能で、新しいバージョンとしてログできます。 



**Returns:**
  `Artifact` オブジェクト。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 

---

### <kbd>method</kbd> `Artifact.new_file`

```python
new_file(
    name: 'str',
    mode: 'str' = 'x',
    encoding: 'str | None' = None
) → Iterator[IO]
```

新しい一時ファイルを開き、アーティファクトに追加します。 



**Args:**
 
 - `name`:  アーティファクトに追加する新しいファイルの名前。 
 - `mode`:  新しいファイルを開くときのファイルアクセスモード。 
 - `encoding`:  新しいファイルを開くときに使用するエンコーディング。 



**Returns:**
 書き込み可能な新しいファイルオブジェクト。クローズ時に自動的にアーティファクトへ追加されます。 



**Raises:**
 
 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため、変更できません。代わりに新しいアーティファクトバージョンをログしてください。 

---

### <kbd>method</kbd> `Artifact.remove`

```python
remove(item: 'StrPath | ArtifactManifestEntry') → None
```

アーティファクトからアイテムを削除します。 



**Args:**
 
 - `item`:  削除するアイテム。特定のマニフェストエントリ、またはアーティファクト相対パスの名前を指定できます。アイテムがディレクトリーに一致する場合、そのディレクトリー内のすべてのアイテムが削除されます。 



**Raises:**
 
 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため、変更できません。代わりに新しいアーティファクトバージョンをログしてください。 
 - `FileNotFoundError`:  アイテムがアーティファクト内に見つからない場合。 

---

### <kbd>method</kbd> `Artifact.save`

```python
save(
    project: 'str | None' = None,
    settings: 'wandb.Settings | None' = None
) → None
```

アーティファクトに対して行った変更を永続化します。 

現在 run の最中であれば、その run がこのアーティファクトをログします。run の最中でない場合は、このアーティファクトを追跡するために、タイプが "auto" の run が作成されます。 



**Args:**
 
 - `project`:  まだ run がコンテキストにない場合に、このアーティファクトに使用するプロジェクト。 
 - `settings`:  自動 run を初期化する際に使用する settings オブジェクト。主にテストハーネスで使用されます。 

---

### <kbd>method</kbd> `Artifact.unlink`

```python
unlink() → None
```

このアーティファクトが現在、昇格されたアーティファクトのコレクションのメンバーである場合、そのリンクを解除します。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 
 - `ValueError`:  アーティファクトがリンクされていない（つまり、ポートフォリオコレクションのメンバーではない）場合。 

---

### <kbd>method</kbd> `Artifact.used_by`

```python
used_by() → list[Run]
```

このアーティファクトと、それにリンクされたアーティファクトを使用した run の一覧を取得します。 



**Returns:**
  `Run` オブジェクトのリスト。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 

---

### <kbd>method</kbd> `Artifact.verify`

```python
verify(root: 'str | None' = None) → None
```

アーティファクトの内容がマニフェストと一致することを検証します。 

ディレクトリー内のすべてのファイルに対してチェックサムを計算し、その値をアーティファクトのマニフェストと照合します。参照は検証されません。 



**Args:**
 
 - `root`:  検証するディレクトリー。None の場合、アーティファクトは './artifacts/self.name/' にダウンロードされます。 



**Raises:**
 
 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。 
 - `ValueError`:  検証に失敗した場合。 

---

### <kbd>method</kbd> `Artifact.wait`

```python
wait(timeout: 'int | None' = None) → Artifact
```

必要に応じて、このアーティファクトのログ完了を待ちます。 



**Args:**
 
 - `timeout`:  待機時間（秒）。 



**Returns:**
 `Artifact` オブジェクト。