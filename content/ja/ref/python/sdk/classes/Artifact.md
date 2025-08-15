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
データセットやモデルのバージョン管理のための柔軟かつ軽量なビルディングブロックです。

空の W&B Artifact を作成します。内容は `add` で始まる各種メソッドで追加します。すべてのファイルを追加し終えたら、`run.log_artifact()` を呼び出してログします。



**引数:**
 
 - `name` (str):  Artifact のわかりやすい名前。W&B アプリ UI やプログラム内で Artifact を識別するための名前です。`use_artifact` パブリック API でインタラクティブに参照可能。この名前にはアルファベット、数字、アンダースコア、ハイフン、ドットが使えます。同じ Project 内でユニークである必要があります。
 - `type` (str):  Artifact のタイプ。Artifact を整理・区別するためのタイプ名。アルファベット・数字・アンダースコア・ハイフン・ドットが使用可能です。一般的には `dataset` や `model` など。モデルを Model Registry に紐付けたい場合は `model` をタイプ名に含めてください。一部は内部利用で予約されていて（例: `job` や `wandb-` で始まるもの）ユーザーが指定できません。
 - `description (str | None) = None`:  Artifact の説明。モデルやデータセット Artifact の場合、モデル・データセットカードのドキュメント記載に活用できます。説明文は W&B アプリで markdown として表示されます。
 - `metadata (dict[str, Any] | None) = None`:  Artifact 付随情報（メタデータ）。キー・バリュー形式の辞書で最大100個までキー指定が可能です。
 - `incremental`:  既存 Artifact を変更する場合は `Artifact.new_draft()` メソッドを使ってください。
 - `use_as`:  非推奨。
 - `is_link`:  この Artifact がリンク Artifact（`True`）かソース Artifact（`False`）かのフラグです。



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

Artifact バージョンに割り当てられている意味のある参照名、 いわゆる「ニックネーム」のリスト。

エイリアスはプログラム的に参照できる可変参照です。W&B アプリまたはプログラムから変更可能。詳細は [新しい artifact バージョンの作成方法](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version)を参照してください。

---

### <kbd>property</kbd> Artifact.collection

この Artifact が取得されたコレクション。

コレクションは Artifact バージョンの順序あるグループです。ポートフォリオまたはリンクコレクションから取得された場合、そのコレクションが返ります。Artifact の発祥元のコレクションは「ソースシーケンス」と呼ばれます。

---

### <kbd>property</kbd> Artifact.commit_hash

この Artifact がコミットされたときに返されるハッシュ値。

---

### <kbd>property</kbd> Artifact.created_at

Artifact 作成日時。

---

### <kbd>property</kbd> Artifact.description

Artifact の説明。

---

### <kbd>property</kbd> Artifact.digest

Artifact の論理ダイジェスト。

ダイジェストは Artifact 内容のチェックサムです。最新バージョンと同じダイジェストの場合、`log_artifact` は何も実行しません（no-op）。

---


### <kbd>property</kbd> Artifact.entity

Artifact コレクションが所属するエンティティ名。

Artifact がリンクの場合、リンク元 Artifact のエンティティ名となります。

---

### <kbd>property</kbd> Artifact.file_count

ファイル（参照を含む）の数。

---

### <kbd>property</kbd> Artifact.history_step

Artifact のソース run でヒストリメトリクスが記録された最も近いステップ。



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

Artifact の ID。

---


### <kbd>property</kbd> Artifact.is_link

この Artifact がリンク Artifact かどうかのフラグ。

True: ソース Artifact へのリンク Artifact。False: ソース Artifact そのもの。

---

### <kbd>property</kbd> Artifact.linked_artifacts

ソース Artifact の全リンク Artifact のリストを返します。

この Artifact 自体がリンク Artifact（`artifact.is_link == True`）の場合は空リストを返します。最大500件まで。

---

### <kbd>property</kbd> Artifact.manifest

Artifact のマニフェスト。

マニフェストには内容の全リストが記載され、Artifact がログされた後は変更できません。

---

### <kbd>property</kbd> Artifact.metadata

ユーザー定義 Artifact メタデータ。

Artifact に紐付いた構造化データ。

---

### <kbd>property</kbd> Artifact.name

Artifact の名前とバージョン。

`{collection}:{alias}`形式の文字列。ログ/保存前の場合はエイリアスなし。リンク Artifact の場合はリンク元 Artifact の名前となります。

---

### <kbd>property</kbd> Artifact.project

Artifact コレクションが所属するプロジェクト名。

Artifact がリンクの場合は、リンク先のプロジェクト名となります。

---

### <kbd>property</kbd> Artifact.qualified_name

Artifact の entity/project/name。

Artifact がリンクの場合は、リンク元 Artifact の qualified name となります。

---

### <kbd>property</kbd> Artifact.size

Artifact の合計サイズ（バイト単位）。

この Artifact がトラッキングする全参照が含まれます。

---

### <kbd>property</kbd> Artifact.source_artifact

ソース Artifact を返します。ソース Artifact とは最初にログされたものです。

自身がソース Artifact（`artifact.is_link == False`）の場合は自身を返します。

---

### <kbd>property</kbd> Artifact.source_collection

Artifact のソースコレクション。

Artifact がどのコレクションからログされたか。

---

### <kbd>property</kbd> Artifact.source_entity

ソース Artifact の entity 名。

---

### <kbd>property</kbd> Artifact.source_name

ソース Artifact の名前とバージョン。

`{source_collection}:{alias}`形式文字列。保存前はバージョンが未確定なので名前のみ。

---

### <kbd>property</kbd> Artifact.source_project

ソース Artifact のプロジェクト名。

---

### <kbd>property</kbd> Artifact.source_qualified_name

ソース Artifact の source_entity/source_project/source_name。

---

### <kbd>property</kbd> Artifact.source_version

ソース Artifact のバージョン。

`v{数字}`形式の文字列。

---

### <kbd>property</kbd> Artifact.state

Artifact の状態。 "PENDING", "COMMITTED", "DELETED" のいずれか。

---

### <kbd>property</kbd> Artifact.tags

この Artifact バージョンにつけられたタグのリスト。

---

### <kbd>property</kbd> Artifact.ttl

Artifact の TTL（Time-To-Live）ポリシー。

TTL ポリシーの期間が過ぎると Artifact は削除されます。`None` の場合は TTL ポリシーを無効化し削除対象になりません。チーム管理者がデフォルト TTL を設定しており Artifact に個別設定がない場合は、そのチーム標準のポリシーを継承します。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がまだログ／保存されていない場合に TTL 継承の取得に失敗。



---

### <kbd>property</kbd> Artifact.type

Artifact のタイプ。典型的には `dataset` や `model` など。

---

### <kbd>property</kbd> Artifact.updated_at

Artifact が最後に更新された時刻。

---

### <kbd>property</kbd> Artifact.url

Artifact の URL を構築します。



**戻り値:**
 
 - `str`:  Artifact の URL。

---

### <kbd>property</kbd> Artifact.use_as

非推奨。

---

### <kbd>property</kbd> Artifact.version

Artifact のバージョン。

`v{数字}`形式の文字列。リンク Artifact の場合はリンク先コレクションのバージョンが表示されます。



---

### <kbd>method</kbd> `Artifact.add`

```python
add(
    obj: 'WBValue',
    name: 'StrPath',
    overwrite: 'bool' = False
) → ArtifactManifestEntry
```

wandb.WBValue `obj` を Artifact に追加します。



**引数:**
 
 - `obj`:  追加したいオブジェクト。Bokeh、JoinedTable、PartitionedTable、Table、Classes、ImageMask、BoundingBoxes2D、Audio、Image、Video、Html、Object3D のいずれかに対応。
 - `name`:  Artifact 内でのオブジェクト追加パス。
 - `overwrite`:  True にすると同一パスに既存ファイルがあれば上書きします。



**戻り値:**
 追加されたマニフェストエントリ



**発生する例外:**
 
 - `ArtifactFinalizedError`:  このバージョンの Artifact は確定済みのため変更できません。新しいバージョンの Artifact をログしてください。

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

ローカルディレクトリを Artifact に追加します。



**引数:**
 
 - `local_path`:  ローカルディレクトリのパス。
 - `name`:  Artifact 内でのサブディレクトリ名。指定がない場合は Artifact ルートへ追加。
 - `skip_cache`:  True にするとアップロード時にキャッシュへのコピー/移動を行いません。
 - `policy`:  デフォルトは "mutable"。
    - mutable: アップロード中の破損を防ぐため、一時的なコピーを作成。
    - immutable: 保護を行わず、ユーザーがファイル削除・変更しないことを前提。
 - `merge`:  False（デフォルト）の場合、既に追加済みで内容が異なるファイルはエラー。True の場合、変更された内容で上書き、新しいファイルは常に追加、既存ファイルは削除されません。ディレクトリ全体の差し替えには `add_dir(local_path, name=my_prefix)` で名前を指定し `remove(my_prefix)` で削除後、再度追加する形を推奨。



**発生する例外:**
 
 - `ArtifactFinalizedError`:  このバージョンの Artifact は確定済みのため変更できません。新しいバージョンをログしてください。
 - `ValueError`:  policy は "mutable" または "immutable" のいずれかでなければなりません。

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

ローカルファイルを Artifact に追加します。



**引数:**
 
 - `local_path`:  追加するファイルのパス。
 - `name`:  Artifact 内で使うファイルのパス。指定がない場合はファイルのベース名。
 - `is_tmp`:  True の場合、競合回避のため決定論的にファイル名がリネームされます。
 - `skip_cache`:  True の場合、アップロード後キャッシュへファイルコピーしません。
 - `policy`:  デフォルトは "mutable"。mutable の場合、アップロード中の破損を防ぐ一時的コピーを作成。immutable の場合は保護を行わずファイル削除や変更がないことを前提とします。
 - `overwrite`:  True の場合、既存ファイルがあれば上書きします。



**戻り値:**
 追加されたマニフェストエントリ。



**発生する例外:**
 
 - `ArtifactFinalizedError`:  このバージョンの Artifact は確定済みのため変更できません。新しいバージョンの Artifact をログしてください。
 - `ValueError`:  policy は "mutable" または "immutable" のいずれかでなければなりません。

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

URI で指定された参照を Artifact に追加します。

ファイルやディレクトリとは異なり、参照は W&B にアップロードされません。詳細は[外部ファイルのトラッキング](https://docs.wandb.ai/guides/artifacts/track-external-files)を参照してください。

デフォルトで以下のスキームがサポートされています:


- http(s): ファイルサイズおよびダイジェストはサーバーから返る `Content-Length` と `ETag` ヘッダーで判定。
- s3: チェックサム・サイズはオブジェクトのメタデータから取得。バケットのバージョン管理有効時はバージョンIDも追跡。
- gs: チェックサム・サイズはオブジェクトのメタデータから取得。バケットバージョン管理有効時はバージョンIDも。
- https、ドメイン `*.blob.core.windows.net` も対応。
- Azure: blob メタデータからチェックサム・サイズ取得。ストレージアカウントのバージョン管理有効時はバージョンIDも追跡。
- file: ファイルシステムからチェックサム・サイズ取得。NFS 共有や外部マウントボリュームのトラッキングなどで便利。

その他のスキームは、ダイジェストは URI のハッシュ、サイズは空白となります。



**引数:**
 
 - `uri`:  追加する参照の URI パス。`Artifact.get_entry` の返り値オブジェクトを指定し、別 Artifact のエントリ参照にも使えます。
 - `name`:  この参照内容を配置する Artifact 内のパス。
 - `checksum`:  参照先リソースのチェックサム計算を行うかどうか。チェックサムは完全性検証を自動化するため推奨です。無効化すると Artifact 作成スピードは向上しますが、ディレクトリの参照追加時に中身を再帰的にたどらなくなるため、実際に中身のファイルが保存されません。参照追加時のみ `checksum=False` にするのが推奨です。
 - `max_objects`:  バケットやディレクトリ参照追加時にインポートする最大オブジェクト数。Amazon S3, GCS, Azure, ローカルファイルはデフォルトで 10,000,000。その他 URI スキーマの上限はありません。



**戻り値:**
 追加されたマニフェストエントリ群。



**発生する例外:**
 
 - `ArtifactFinalizedError`:  このバージョンの Artifact は確定済みのため変更できません。新しいバージョンの Artifact をログしてください。

---

### <kbd>method</kbd> `Artifact.checkout`

```python
checkout(root: 'str | None' = None) → str
```

指定したルートディレクトリを Artifact の内容で置き換えます。

警告: Artifact に含まれない `root` 以下の全ファイルは削除されます。



**引数:**
 
 - `root`:  この Artifact のファイルで置き換えるディレクトリ。



**戻り値:**
 チェックアウトされた内容のパス。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。

---

### <kbd>method</kbd> `Artifact.delete`

```python
delete(delete_aliases: 'bool' = False) → None
```

Artifact およびそのファイルを削除します。

リンク Artifact で実行した場合はリンクのみ削除され、ソース Artifact は削除されません。

ソース Artifact とリンク Artifact の紐付けを解除するには `artifact.unlink()` を使ってください。



**引数:**
 
 - `delete_aliases`:  True の場合は Artifact に関連するすべてのエイリアスも削除。False の場合、既存のエイリアスがあると例外が発生します。Artifact がリンクの場合はこのパラメータは無視されます。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。

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

Artifact の内容を指定ディレクトリにダウンロードします。

既存のファイルはそのまま残ります。`root` 以下の内容を Artifact と完全一致させたい場合は、事前に `root` を削除してください。



**引数:**
 
 - `root`:  W&B が Artifact ファイルを格納するディレクトリ。
 - `allow_missing_references`:  True の場合、不正な参照パスは無視してファイルダウンロードを継続します。
 - `skip_cache`:  True の場合、キャッシュをスキップし、各ファイルを指定ディレクトリ直下へ直接ダウンロードします。
 - `path_prefix`:  指定があれば、与えたパスプレフィックスで始まるファイルのみダウンロード。UNIX 形式（スラッシュ区切り）。
 - `multipart`:  None（デフォルト）の場合、2GB 超のファイルは並列マルチパートダウンロード。それ以外の場合 True/False で並列・直列ダウンロードを強制できます。



**戻り値:**
 ダウンロードされた内容のパス。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。

---

### <kbd>method</kbd> `Artifact.file`

```python
file(root: 'str | None' = None) → StrPath
```

単一ファイルの Artifact を指定ディレクトリ（`root`）へダウンロードします。



**引数:**
 
 - `root`:  ファイルを保存するディレクトリ。デフォルトは `./artifacts/self.name/` です。



**戻り値:**
 ダウンロードされたファイルのフルパス。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。
 - `ValueError`:  Artifact に複数ファイルが含まれる場合。

---

### <kbd>method</kbd> `Artifact.files`

```python
files(names: 'list[str] | None' = None, per_page: 'int' = 50) → ArtifactFiles
```

この Artifact に保存されているすべてのファイルを順にイテレートします。



**引数:**
 
 - `names`:  Artifact ルートからの相対パスでリストアップしたいファイル名。
 - `per_page`:  1リクエストあたり返すファイル数。



**戻り値:**
 `File` オブジェクトのイテレータ。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。

---

### <kbd>method</kbd> `Artifact.finalize`

```python
finalize() → None
```

Artifact バージョンを確定します。

一度確定したバージョンは変更できません。追加のデータを記録するには新規バージョンを作成してください。Artifact を `log_artifact` でログした時点で自動で確定されます。

---

### <kbd>method</kbd> `Artifact.get`

```python
get(name: 'str') → WBValue | None
```

Artifact 相対 `name` の WBValue オブジェクトを取得します。



**引数:**
 
 - `name`:  取得したい Artifact 内の相対パス名。



**戻り値:**
 `run.log()` および W&B UI で可視化できる W&B オブジェクト。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact が未ログまたは run がオフラインの場合。

---

### <kbd>method</kbd> `Artifact.get_added_local_path_name`

```python
get_added_local_path_name(local_path: 'str') → str | None
```

ローカルのファイルパスから追加された Artifact 内相対名を取得します。



**引数:**
 
 - `local_path`:  Artifact 内相対名へ変換するローカルパス。



**戻り値:**
 変換された Artifact 内相対名。

---

### <kbd>method</kbd> `Artifact.get_entry`

```python
get_entry(name: 'StrPath') → ArtifactManifestEntry
```

指定した名前のエントリを取得します。



**引数:**
 
 - `name`:  取得したい Artifact 内相対名



**戻り値:**
 `W&B` オブジェクト。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact が未ログまたは run がオフラインの場合。
 - `KeyError`:  指定した名前のエントリが存在しない場合。

---

### <kbd>method</kbd> `Artifact.get_path`

```python
get_path(name: 'StrPath') → ArtifactManifestEntry
```

非推奨です。`get_entry(name)` をご利用ください。

---

### <kbd>method</kbd> `Artifact.is_draft`

```python
is_draft() → bool
```

Artifact が未保存かどうか確認します。



**戻り値:**
  Boolean 値。保存済みなら False、未保存なら True。

---

### <kbd>method</kbd> `Artifact.json_encode`

```python
json_encode() → dict[str, Any]
```

Artifact を JSON 形式にエンコードした内容を返します。



**戻り値:**
  文字列型のキーで属性を表した `dict`。

---

### <kbd>method</kbd> `Artifact.link`

```python
link(target_path: 'str', aliases: 'list[str] | None' = None) → Artifact | None
```

この Artifact をポートフォリオ（アーティファクトのプロモートコレクション）へリンクします。



**引数:**
 
 - `target_path`:  プロジェクト内でのポートフォリオのパス。スキーマは `{portfolio}`、`{project}/{portfolio}` または `{entity}/{project}/{portfolio}` のいずれか。Model Registry へのリンクは `{"model-registry"}/{Registered Model Name}` または `{entity}/{"model-registry"}/{Registered Model Name}` の形式で指定。
 - `aliases`:  指定ポートフォリオ内で Artifact を一意に識別するための文字列リスト。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。



**戻り値:**
 リンクに成功した場合はリンク Artifact、失敗時は None。

---

### <kbd>method</kbd> `Artifact.logged_by`

```python
logged_by() → Run | None
```

この Artifact を最初にログした W&B run を取得します。



**戻り値:**
  この Artifact を最初にログした W&B run の名前。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。

---

### <kbd>method</kbd> `Artifact.new_draft`

```python
new_draft() → Artifact
```

この確定済み Artifact と同じ内容で新たなドラフト Artifact を作成します。

既存 Artifact の編集は「インクリメンタル Artifact」とも呼ばれる新規バージョン作成となります。返り値の Artifact は追加や変更が可能で新バージョンとしてログできます。



**戻り値:**
  `Artifact` オブジェクト。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。

---

### <kbd>method</kbd> `Artifact.new_file`

```python
new_file(
    name: 'str',
    mode: 'str' = 'x',
    encoding: 'str | None' = None
) → Iterator[IO]
```

一時ファイルを新規作成し Artifact に追加します。



**引数:**
 
 - `name`:  Artifact に追加する新規ファイル名。
 - `mode`:  ファイルオープンに使うアクセスモード。
 - `encoding`:  ファイルオープン時のエンコーディング。



**戻り値:**
 書き込み可能なファイルオブジェクト。クローズ時に自動的に Artifact に追加されます。



**発生する例外:**
 
 - `ArtifactFinalizedError`:  このバージョンの Artifact は確定済みのため変更できません。新しいバージョンをログしてください。

---

### <kbd>method</kbd> `Artifact.remove`

```python
remove(item: 'StrPath | ArtifactManifestEntry') → None
```

Artifact からアイテムを削除します。



**引数:**
 
 - `item`:  削除対象。特定マニフェストエントリまたは Artifact 内相対パス名。ディレクトリが一致した場合は配下すべてのアイテムが削除されます。



**発生する例外:**
 
 - `ArtifactFinalizedError`:  このバージョンの Artifact は確定済みのため変更できません。新しいバージョンをログしてください。
 - `FileNotFoundError`:  指定アイテムが Artifact に存在しない場合。

---

### <kbd>method</kbd> `Artifact.save`

```python
save(
    project: 'str | None' = None,
    settings: 'wandb.Settings | None' = None
) → None
```

Artifact に施した変更を保存します。

現在 run で実行中の場合はその run で Artifact がログされます。run がない場合は "auto" 型の run が生成されて Artifact のトラッキングを行います。



**引数:**
 
 - `project`:  run が存在しない場合にこの Artifact 用に利用するプロジェクト名。
 - `settings`:  自動 run 初期化時に利用する Settings オブジェクト（主にテスト時に利用）。

---

### <kbd>method</kbd> `Artifact.unlink`

```python
unlink() → None
```

この Artifact がプロモートコレクションのメンバーである場合、そのリンクを解除します。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。
 - `ValueError`:  Artifact がリンクされていない（＝ポートフォリオコレクションのメンバーでない）場合。

---

### <kbd>method</kbd> `Artifact.used_by`

```python
used_by() → list[Run]
```

この Artifact および関連リンク Artifact を使用した run 一覧を取得します。



**戻り値:**
  `Run` オブジェクト一覧。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。

---

### <kbd>method</kbd> `Artifact.verify`

```python
verify(root: 'str | None' = None) → None
```

Artifact の内容がマニフェストと一致するか検証します。

ディレクトリ内のすべてのファイルがチェックサム計算され Artifact のマニフェストと照合されます。参照タイプは検証されません。



**引数:**
 
 - `root`:  検証するディレクトリ。None の場合は `./artifacts/self.name/` にダウンロードして検証。



**発生する例外:**
 
 - `ArtifactNotLoggedError`:  Artifact がログされていない場合。
 - `ValueError`:  検証に失敗した場合。

---

### <kbd>method</kbd> `Artifact.wait`

```python
wait(timeout: 'int | None' = None) → Artifact
```

必要に応じて、この Artifact のロギング完了まで待機します。



**引数:**
 
 - `timeout`:  待機時間（秒）。



**戻り値:**
 `Artifact` オブジェクト。
