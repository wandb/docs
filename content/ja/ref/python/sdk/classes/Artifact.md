---
title: アーティファクト
object_type: python_sdk_actions
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/artifacts/artifact.py >}}




## <kbd>class</kbd> `Artifact`
データセットやモデルのバージョン管理のための柔軟で軽量な構成要素です。

空の W&B Artifact を作成します。アーティファクトの中身は `add` で始まるメソッドを使って追加してください。必要なファイルをすべて追加したら、`run.log_artifact()` を呼び出してアーティファクトをログできます。



**引数:**

 - `name` (str):  アーティファクトの人間が読める名前。W&B アプリ UI やプログラム上で特定のアーティファクトを識別するために使います。Public API の `use_artifact` を使ってインタラクティブに参照できます。英字、数字、アンダースコア、ハイフン、ドットが使えます。プロジェクト内で一意な名前にしてください。
 - `type` (str):  アーティファクトのタイプ。タイプでアーティファクトを整理・区別できます。任意の英数・アンダースコア・ハイフン・ドットを含む文字列が使えます。一般的なタイプには `dataset` や `model` があります。`model` をタイプ文字列に含めると、W&B Model Registry と関連付けられます。一部のタイプ（`job` や `wandb-` で始まるもの）は内部用で、ユーザーは指定できません。
 - `description (str | None) = None`:  アーティファクトの説明。モデルやデータセット用アーティファクトの場合、標準化されたチーム用のモデルカードやデータセットカードをドキュメントとして追加できます。`Artifact.description` 属性、または W&B アプリ UI からプログラム上で説明を確認できます。アプリ上では markdown として表示されます。
 - `metadata (dict[str, Any] | None) = None`:  アーティファクトに関する追加情報。キーと値のペアからなる辞書形式で指定します。最大 100 キーまで設定可能です。
 - `incremental`:  既存のアーティファクトを変更したい場合は `Artifact.new_draft()` メソッドを使用してください。
 - `use_as`:  廃止予定。
 - `is_link`:  アーティファクトがリンクアーティファクト（`True`）か、ソースアーティファクト（`False`）か示すブール値。



**戻り値:**
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

アーティファクトバージョンに割り当てられる、わかりやすい参照名や「ニックネーム」のリスト。

エイリアスはプログラムから参照・変更可能です。W&B アプリまたはプログラム上でアーティファクトのエイリアスを変更できます。詳細は [Create new artifact versions](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version) をご覧ください。

---

### <kbd>property</kbd> Artifact.collection

このアーティファクトが取得されたコレクション。

コレクションはアーティファクトのバージョンの順序付きのグループです。ポートフォリオやリンクコレクションから取得した場合は、そのコレクションが返されます。アーティファクトが元々所属したコレクションは「ソースシーケンス」と呼ばれます。

---

### <kbd>property</kbd> Artifact.commit_hash

このアーティファクトがコミットされたときに返されるハッシュ値。

---

### <kbd>property</kbd> Artifact.created_at

アーティファクトが作成されたタイムスタンプ。

---

### <kbd>property</kbd> Artifact.description

アーティファクトの説明。

---

### <kbd>property</kbd> Artifact.digest

アーティファクトの論理的ダイジェスト（チェックサム）。

このダイジェストはアーティファクト内容のチェックサムです。もし現行の `latest` バージョンと同じダイジェストの場合、`log_artifact` は何もしません。

---


### <kbd>property</kbd> Artifact.entity

アーティファクトコレクションが所属するエンティティ（entity）の名前。

アーティファクトがリンクの場合は、リンク先のアーティファクトの entity になります。

---

### <kbd>property</kbd> Artifact.file_count

ファイルの数（参照ファイル含む）。

---

### <kbd>property</kbd> Artifact.history_step

アーティファクトのソース run で履歴メトリクスが記録された最近のステップ。



**例:**
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

アーティファクトがリンクアーティファクトかどうかを示すブール値。

True の場合: アーティファクトはソースアーティファクトへのリンクです。False の場合: ソースアーティファクトです。

---

### <kbd>property</kbd> Artifact.linked_artifacts

ソースアーティファクトにリンクされた全てのアーティファクトのリストを返します。

アーティファクトがリンクアーティファクト（`artifact.is_link == True`）の場合は空リストを返します。最大 500 件までです。

---

### <kbd>property</kbd> Artifact.manifest

アーティファクトのマニフェスト。

すべての内容がリストされており、アーティファクトをログした後は変更できません。

---

### <kbd>property</kbd> Artifact.metadata

ユーザー定義のアーティファクトメタデータ。

アーティファクトに紐づく構造化データ。

---

### <kbd>property</kbd> Artifact.name

アーティファクトの名前とバージョン。

`{collection}:{alias}` 形式の文字列です。アーティファクトをログまたは保存する前は alias を含みません。リンクの場合はリンク先アーティファクトの名前になります。

---

### <kbd>property</kbd> Artifact.project

アーティファクトコレクションが属するプロジェクト名。

リンクの場合は、リンク先アーティファクトのプロジェクトになります。

---

### <kbd>property</kbd> Artifact.qualified_name

アーティファクトの entity/project/name。

リンクの場合は、リンク先アーティファクトのパスの qualified name になります。

---

### <kbd>property</kbd> Artifact.size

アーティファクトの合計サイズ（バイト単位）。

このアーティファクトで参照されているファイルも含みます。

---

### <kbd>property</kbd> Artifact.source_artifact

ソースアーティファクトを返します。ソースアーティファクトとは、元々ログされたアーティファクトです。

自身がソースアーティファクト（`artifact.is_link == False`）の場合、自分自身を返します。

---

### <kbd>property</kbd> Artifact.source_collection

アーティファクトのソースコレクション。

このコレクションはアーティファクトがログされた元のコレクションです。

---

### <kbd>property</kbd> Artifact.source_entity

ソースアーティファクトの entity 名。

---

### <kbd>property</kbd> Artifact.source_name

ソースアーティファクトの名前とバージョン。

`{source_collection}:{alias}` という形式の文字列です。アーティファクト保存前は名前のみ（バージョン未定）です。

---

### <kbd>property</kbd> Artifact.source_project

ソースアーティファクトのプロジェクト名。

---

### <kbd>property</kbd> Artifact.source_qualified_name

ソースアーティファクトの source_entity/source_project/source_name。

---

### <kbd>property</kbd> Artifact.source_version

ソースアーティファクトのバージョン。

`v{number}` 形式の文字列です。

---

### <kbd>property</kbd> Artifact.state

アーティファクトのステータス。"PENDING"、"COMMITTED"、"DELETED" のいずれか。

---

### <kbd>property</kbd> Artifact.tags

このアーティファクトバージョンに割り当てられたタグ（1つ以上）のリスト。

---

### <kbd>property</kbd> Artifact.ttl

アーティファクトの TTL（Time-To-Live、保持期間）ポリシー。

TTL ポリシーの期間が経過するとアーティファクトは速やかに削除されます。`None` を指定すると TTL ポリシーは無効となり、チーム既定 TTL があってもスケジュール削除されません。チーム管理者が既定 TTL を定義している場合でアーティファクトに個別設定がなければ、チームの既定 TTL を継承します。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログ・保存されていない場合、継承 TTL の取得に失敗します。

---

### <kbd>property</kbd> Artifact.type

アーティファクトのタイプ。一般的なタイプには `dataset` や `model` など。

---

### <kbd>property</kbd> Artifact.updated_at

アーティファクトが最後に更新された日時。

---

### <kbd>property</kbd> Artifact.url

アーティファクトの URL を作成します。



**戻り値:**

 - `str`:  アーティファクトの URL。

---

### <kbd>property</kbd> Artifact.use_as

廃止予定。

---

### <kbd>property</kbd> Artifact.version

アーティファクトのバージョン。

`v{number}` 形式の文字列。リンクアーティファクトの場合は、リンク元コレクションのバージョンが使われます。



---

### <kbd>method</kbd> `Artifact.add`

```python
add(
    obj: 'WBValue',
    name: 'StrPath',
    overwrite: 'bool' = False
) → ArtifactManifestEntry
```

wandb.WBValue 型の `obj` をアーティファクトに追加します。



**引数:**

 - `obj`:  追加するオブジェクト。現在は Bokeh、JoinedTable、PartitionedTable、Table、Classes、ImageMask、BoundingBoxes2D、Audio、Image、Video、Html、Object3D のいずれかをサポートします。
 - `name`:  アーティファクト内でオブジェクトを追加するパス。
 - `overwrite`:  True の場合、同じファイルパスが存在する場合は上書きします。



**戻り値:**
 追加されたマニフェストエントリ



**例外:**

 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため変更できません。新しいバージョンをログしてください。

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

ローカルディレクトリーをアーティファクトに追加します。



**引数:**

 - `local_path`:  ローカルディレクトリーのパス。
 - `name`:  アーティファクト内のサブディレクトリー名。指定した名前は W&B アプリ UI 上でアーティファクトの `type` ごとにネスト表示されます。デフォルトはアーティファクトのルートです。
 - `skip_cache`:  True にすると、アップロード時にファイルのキャッシュコピーを作成しません。
 - `policy`:  標準では "mutable"。
    - mutable: アップロード時の破損を防ぐため、一時コピーを作成します。
    - immutable: 保護を無効にし、ユーザーの責任でファイルの変更/削除を避けます。
 - `merge`:  False（デフォルト）の場合、前回 add_dir で追加済みかつ内容が変わっているファイルがあれば ValueError が投げられます。True の場合は変更されたファイルが上書きされます。新しいファイルのみ追加され、削除は行いません。ディレクトリー全体を置き換える場合は、`add_dir(local_path, name=my_prefix)` で追加し `remove(my_prefix)` で削除し、再度追加してください。



**例外:**

 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため変更できません。新しいバージョンをログしてください。
 - `ValueError`:  policy は "mutable" か "immutable" のみ利用できます。

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



**引数:**

 - `local_path`:  追加するファイルのパス。
 - `name`:  アーティファクト内で使うパス。指定しない場合はファイルベース名になります。
 - `is_tmp`:  True にするとファイルが決定論的にリネームされ衝突を防ぎます。
 - `skip_cache`:  True でアップロード後ファイルキャッシュをスキップします。
 - `policy`:  デフォルトは "mutable"。mutable ならアップロード中の破損防止用に一時コピーを作成します。immutable なら保護せずユーザー責任です。
 - `overwrite`:  True で既存ファイルを上書きします。



**戻り値:**
 追加されたマニフェストエントリ。



**例外:**

 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため変更できません。新しいバージョンをログしてください。
 - `ValueError`:  policy は "mutable" または "immutable" が必要です。

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

ファイルやディレクトリーとは異なり、参照は W&B にアップロードされません。詳細は [外部ファイルのトラッキング](https://docs.wandb.ai/guides/artifacts/track-external-files) を参照してください。

デフォルトでサポートしているスキームは以下のとおりです:


- http(s): サーバーの `Content-Length` と `ETag` 応答ヘッダーからサイズ・ダイジェストを取得します。
- s3: オブジェクトのメタデータからチェックサムとサイズを取得します。バケットのバージョン管理が有効ならバージョン ID も追跡。
- gs: メタデータからチェックサムとサイズを取得し、バケットバージョン管理が有効ならバージョン ID も取得。
- https, ドメインが `*.blob.core.windows.net` に一致
- Azure: blob メタデータからチェックサムとサイズ、バージョン管理が有効ならバージョン ID も追跡。
- file: ファイルシステムからチェックサムとサイズを取得。NFS 共有や外部マウントボリュームでアップロードせずにトラッキングしたい場合に便利です。

その他のスキームでは、ダイジェストは URI のハッシュ、サイズは空のままです。



**引数:**

 - `uri`:  参照を追加する URI パス。`Artifact.get_entry` から得られるオブジェクトも指定可能です（他のアーティファクトのエントリ参照）。
 - `name`:  この参照の内容を格納するアーティファクト内パス。
 - `checksum`:  参照先リソースのチェックサムを計算するかどうか。推奨は True（整合性検証のため）。False にすると作成が速くなりますが、参照ディレクトリーは再帰されず中身は保存されません。参照オブジェクトを追加する際は `checksum=False` の使用を推奨し、この場合 URI が変わった時だけ新しいバージョンが作成されます。
 - `max_objects`:  参照がディレクトリーやバケットのプレフィックスの場合、最大で何個オブジェクトを扱うか。Amazon S3, GCS, Azure, ローカルファイルのデフォルト最大は 1,000 万。それ以外の URI では上限はありません。



**戻り値:**
 追加されたマニフェストエントリ群。



**例外:**

 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため変更できません。新しいバージョンをログしてください。

---

### <kbd>method</kbd> `Artifact.checkout`

```python
checkout(root: 'str | None' = None) → str
```

指定したルートディレクトリーをアーティファクトの内容で置換します。

警告: `root` に含まれるアーティファクト外の全ファイルは削除されます。



**引数:**

 - `root`:  このアーティファクトのファイルで置換するディレクトリー。



**戻り値:**
 チェックアウトした内容のパス。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。

---

### <kbd>method</kbd> `Artifact.delete`

```python
delete(delete_aliases: 'bool' = False) → None
```

アーティファクトとそのファイルを削除します。

リンクアーティファクトに対して呼び出した場合は、リンクのみ削除されソースアーティファクトは影響を受けません。

ソースアーティファクトとリンクアーティファクトの間のリンクのみを削除したい場合は `artifact.unlink()` を使ってください。



**引数:**

 - `delete_aliases`:  True にすると、アーティファクトに付随するすべてのエイリアスを削除します。False の場合、既存エイリアスがあると例外が発生します。リンクアーティファクトの場合はこのパラメータは無視されます。



**例外:**

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

アーティファクトの内容を指定したルートディレクトリーへダウンロードします。

`root` に存在する既存ファイルは変更されません。内容を完全に一致させたい場合はダウンロード前に `root` ディレクトリ自体を削除してください。



**引数:**

 - `root`:  W&B がアーティファクトのファイルを保存するディレクトリー。
 - `allow_missing_references`:  True にすると、無効な参照パスがあっても無視して参照ファイルのダウンロードを続行します。
 - `skip_cache`:  True の場合、ダウンロード時にアーティファクトキャッシュをスキップし、それぞれのファイルをダウンロードディレクトリーまたはデフォルトルートに直接保存します。
 - `path_prefix`:  指定された場合、そのプレフィックスで始まるファイルのみダウンロードします（unix形式、スラッシュ区切り）。
 - `multipart`:  None（デフォルト）の場合、2GB を超えるファイルはマルチパートで並列ダウンロードします。True/False で並列/直列ダウンロードを強制できます。



**戻り値:**
 ダウンロードされた内容のパス。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。

---

### <kbd>method</kbd> `Artifact.file`

```python
file(root: 'str | None' = None) → StrPath
```

単一ファイルのアーティファクトを指定のディレクトリーにダウンロードします。



**引数:**

 - `root`:  ファイルを保存するルートディレクトリー。デフォルトは `./artifacts/self.name/`。



**戻り値:**
 ダウンロードしたファイルのフルパス。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。
 - `ValueError`:  アーティファクト内に複数ファイルが含まれる場合。

---

### <kbd>method</kbd> `Artifact.files`

```python
files(names: 'list[str] | None' = None, per_page: 'int' = 50) → ArtifactFiles
```

このアーティファクトに格納された全ファイルをイテレートします。



**引数:**

 - `names`:  アーティファクトルートからの相対パスでリストしたいファイル名。
 - `per_page`:  1リクエストあたりで返されるファイル数。



**戻り値:**
 `File` オブジェクトを含むイテレータ。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。

---

### <kbd>method</kbd> `Artifact.finalize`

```python
finalize() → None
```

アーティファクトバージョンを確定（finalize）します。

一度 finalize されたアーティファクトバージョンは変更できません。追加データをログしたい場合は新しいバージョンを作成してください。アーティファクトは `log_artifact` で自動的に finalize されます。

---

### <kbd>method</kbd> `Artifact.get`

```python
get(name: 'str') → WBValue | None
```

アーティファクト内の relative `name` にある WBValue オブジェクトを取得。



**引数:**

 - `name`:  取得したいアーティファクト内での相対名。



**戻り値:**
 W&B でログ・可視化できるオブジェクト。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトが未ログ、または run がオフラインの場合。

---

### <kbd>method</kbd> `Artifact.get_added_local_path_name`

```python
get_added_local_path_name(local_path: 'str') → str | None
```

ローカルファイルシステムパスで追加したファイルの、アーティファクト内相対名を取得します。



**引数:**

 - `local_path`:  アーティファクト内相対名に解決したいローカルパス。



**戻り値:**
 アーティファクト内相対名。

---

### <kbd>method</kbd> `Artifact.get_entry`

```python
get_entry(name: 'StrPath') → ArtifactManifestEntry
```

指定した名前のエントリを取得します。



**引数:**

 - `name`:  取得したいアーティファクト内相対名



**戻り値:**
 `W&B` オブジェクト。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトが未ログ、または run がオフラインの場合。
 - `KeyError`:  指定名のエントリが存在しない場合。

---

### <kbd>method</kbd> `Artifact.get_path`

```python
get_path(name: 'StrPath') → ArtifactManifestEntry
```

廃止予定。`get_entry(name)` を利用してください。

---

### <kbd>method</kbd> `Artifact.is_draft`

```python
is_draft() → bool
```

アーティファクトが未保存かどうかを確認します。



**戻り値:**
  ブール値。保存済みの場合 False、未保存の場合 True。

---

### <kbd>method</kbd> `Artifact.json_encode`

```python
json_encode() → dict[str, Any]
```

アーティファクトを JSON 形式でエンコードして返します。



**戻り値:**
  `string` キーで属性を表す `dict`。

---

### <kbd>method</kbd> `Artifact.link`

```python
link(target_path: 'str', aliases: 'list[str] | None' = None) → Artifact | None
```

このアーティファクトをポートフォリオ（昇格コレクション）へリンクします。



**引数:**

 - `target_path`:  プロジェクト内のポートフォリオパス。次のいずれかの形式 `{portfolio}`、`{project}/{portfolio}`、`{entity}/{project}/{portfolio}` で指定します。W&B Model Registry へリンクする場合は、`{"model-registry"}/{Registered Model Name}` または `{entity}/{"model-registry"}/{Registered Model Name}` 形式にします。
 - `aliases`:  指定ポートフォリオ内でアーティファクトを一意に識別するための文字列リスト。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。



**戻り値:**
 リンクに成功した場合はリンク先アーティファクト。失敗時は None。

---

### <kbd>method</kbd> `Artifact.logged_by`

```python
logged_by() → Run | None
```

このアーティファクトを最初にログした W&B run を取得します。



**戻り値:**
  このアーティファクトを最初にログした W&B run の名前。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。

---

### <kbd>method</kbd> `Artifact.new_draft`

```python
new_draft() → Artifact
```

この確定済みアーティファクトの内容を引き継いだ新しいドラフトアーティファクトを作成します。

既存アーティファクトを修正する場合は、「インクリメンタルアーティファクト」として新しいバージョンが作成されます。返されたアーティファクトはさらに拡張・修正ができ、新バージョンとしてログできます。



**戻り値:**
  `Artifact` オブジェクト。



**例外:**

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

新しい一時ファイルを開いてアーティファクトに追加します。



**引数:**

 - `name`:  アーティファクトに追加する新ファイルの名前。
 - `mode`:  ファイルのアクセスモード。
 - `encoding`:  新ファイルを開く際のエンコーディング。



**戻り値:**
 書き込み可能な新ファイルオブジェクト。閉じると自動的にアーティファクトに追加されます。



**例外:**

 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため変更できません。新しいバージョンをログしてください。

---

### <kbd>method</kbd> `Artifact.remove`

```python
remove(item: 'StrPath | ArtifactManifestEntry') → None
```

アーティファクトからアイテムを削除します。



**引数:**

 - `item`:  削除するアイテム。マニフェストエントリまたはアーティファクト相対パス指定可能。ディレクトリーに一致すれば中の全アイテムが削除されます。



**例外:**

 - `ArtifactFinalizedError`:  このアーティファクトバージョンは確定済みのため変更できません。新しいバージョンをログしてください。
 - `FileNotFoundError`:  アイテムが見つからない場合。

---

### <kbd>method</kbd> `Artifact.save`

```python
save(
    project: 'str | None' = None,
    settings: 'wandb.Settings | None' = None
) → None
```

アーティファクトに対する変更を永続化します。

現在 run 内の場合はその run がアーティファクトをログします。run 内でなければ "auto" タイプの run が作成され、そのアーティファクトがトラッキングされます。



**引数:**

 - `project`:  まだ run が存在しない場合にアーティファクトを利用するプロジェクト名。
 - `settings`:  自動 run 初期化時の設定オブジェクト。テスト環境などで利用されます。

---

### <kbd>method</kbd> `Artifact.unlink`

```python
unlink() → None
```

アーティファクトが昇格コレクションのメンバーである場合、そのリンクを解除します。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。
 - `ValueError`:  アーティファクトがリンクされていない場合（ポートフォリオコレクションのメンバーでない場合）。

---

### <kbd>method</kbd> `Artifact.used_by`

```python
used_by() → list[Run]
```

このアーティファクトとそのリンク先アーティファクトを利用した run の一覧を取得します。



**戻り値:**
  `Run` オブジェクトのリスト。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。

---

### <kbd>method</kbd> `Artifact.verify`

```python
verify(root: 'str | None' = None) → None
```

アーティファクトの内容がマニフェストと一致していることを検証します。

ディレクトリー内のすべてのファイルのチェックサムを計算し、その値がマニフェストと一致するかを確認します。参照ファイルは検証しません。



**引数:**

 - `root`:  検証するディレクトリー。None の場合は `'./artifacts/self.name/'` にダウンロードします。



**例外:**

 - `ArtifactNotLoggedError`:  アーティファクトがログされていない場合。
 - `ValueError`:  検証に失敗した場合。

---

### <kbd>method</kbd> `Artifact.wait`

```python
wait(timeout: 'int | None' = None) → Artifact
```

必要に応じてアーティファクトのログ処理完了を待ちます。



**引数:**

 - `timeout`:  待機時間（秒単位）。



**戻り値:**
 `Artifact` オブジェクト。
