---
title: Artifact
menu:
  reference:
    identifier: ja-ref-python-artifact
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L96-L2410 >}}

データセット と モデル の バージョン管理 のための、柔軟 で 軽量 な 構成要素 です。

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

空 の W&B Artifact を構築します。`add` で始まる メソッド で Artifact の コンテンツ を 設定 します。Artifact に 必要な ファイル が 全て 揃ったら、`wandb.log_artifact()` を 呼び出して ログ に 記録 できます。

| arg |  |
| :--- | :--- |
| `name` | Artifact の わかりやすい 名前。この 名前 を 使用して、W&B App UI で 特定 の Artifact を 識別したり、プログラム で 識別したりします。`use_artifact` Public API を 使用して、Artifact をインタラクティブ に 参照 できます。名前 には、文字、数字、アンダースコア、ハイフン、ドット を 含める ことができます。名前 は プロジェクト 全体 で 一意 で ある 必要 が あります。 |
| `type` | Artifact の タイプ。Artifact の タイプ を 使用して、Artifact を 整理 および 区別 します。文字、数字、アンダースコア、ハイフン、ドット を 含む 任意 の 文字列 を 使用 できます。一般的 な タイプ には、`dataset` や `model` があります。Artifact を W&B Model Registry に リンク する 場合 は、タイプ 文字列 に `model` を 含めます。 |
| `description` | Artifact の 説明。Model または Dataset Artifact の 場合 は、標準化 された チームモデル または データセット カード の ドキュメント を 追加 します。`Artifact.description` 属性 を 使用して プログラム で、または W&B App UI を 使用して プログラム で Artifact の 説明 を 表示 します。W&B は、W&B App で 説明 を Markdown として レンダリング します。 |
| `metadata` | Artifact に関する 追加 情報。メタデータ を キー と 値 の ペア の ディクショナリー として 指定 します。合計 100 個 まで の キー を 指定 できます。 |
| `incremental` | 既存 の Artifact を 変更 する には、代わり に `Artifact.new_draft()` メソッド を 使用 します。 |
| `use_as` | W&B Launch 固有 の パラメータ。一般 的 な 使用 には お勧め しません。 |

| 戻り値 |  |
| :--- | :--- |
| `Artifact` オブジェクト。 |

| 属性 |  |
| :--- | :--- |
| `aliases` | Artifact バージョン に 割り当てられた、1 つ または 複数 の 意味的に わかりやすい 参照 または 識別 用 の 「ニックネーム」 の リスト。エイリアス は、プログラム で 参照 できる 可変 参照 です。W&B App UI を 使用するか、プログラム で Artifact の エイリアス を 変更 します。詳細 については、[新しい Artifact バージョン を 作成 する](https://docs.wandb.ai/guides/artifacts/create-a-new-artifact-version) を 参照 して ください。 |
| `collection` | この Artifact が 取得 された コレクション。コレクション は Artifact バージョン の 順序付けられた グループ です。この Artifact が ポートフォリオ / リンクされた コレクション から 取得 された 場合、Artifact バージョン の 元 の コレクション で はなく、その コレクション が 返されます。Artifact の 元 の コレクション は、ソース シーケンス と 呼ばれます。 |
| `commit_hash` | この Artifact が コミット された とき に 返された ハッシュ。 |
| `created_at` | Artifact が 作成 された とき の タイムスタンプ。 |
| `description` | Artifact の 説明。 |
| `digest` | Artifact の 論理 ダイジェスト。ダイジェスト は Artifact の コンテンツ の チェックサム です。Artifact の ダイジェスト が 現在 の `latest` バージョン と 同じ 場合、`log_artifact` は 何も しません。 |
| `entity` | セカンダリ (ポートフォリオ) Artifact コレクション の エンティティ の 名前。 |
| `file_count` | ファイル の 数 (参照 を 含む)。 |
| `id` | Artifact の ID。 |
| `manifest` | Artifact の マニフェスト。マニフェスト には 全て の コンテンツ が リスト されており、Artifact が ログ に 記録 された 後 は 変更 できません。 |
| `metadata` | ユーザー定義 の Artifact メタデータ。Artifact に 関連付けられた 構造化 データ。 |
| `name` | セカンダリ (ポートフォリオ) コレクション 内 の Artifact 名 と バージョン。`{collection}:{alias}` という 形式 の 文字列。Artifact が 保存 される 前 は、バージョン が まだ わからない ため、名前 のみ が 含まれます。 |
| `project` | セカンダリ (ポートフォリオ) Artifact コレクション の プロジェクト の 名前。 |
| `qualified_name` | セカンダリ (ポートフォリオ) コレクション の エンティティ / プロジェクト / 名前。 |
| `size` | Artifact の 合計 サイズ (バイト単位)。この Artifact で 追跡 される 全て の 参照 が 含まれます。 |
| `source_collection` | Artifact の プライマリ (シーケンス) コレクション。 |
| `source_entity` | プライマリ (シーケンス) Artifact コレクション の エンティティ の 名前。 |
| `source_name` | プライマリ (シーケンス) コレクション 内 の Artifact 名 と バージョン。`{collection}:{alias}` という 形式 の 文字列。Artifact が 保存 される 前 は、バージョン が まだ わからない ため、名前 のみ が 含まれます。 |
| `source_project` | プライマリ (シーケンス) Artifact コレクション の プロジェクト の 名前。 |
| `source_qualified_name` | プライマリ (シーケンス) コレクション の エンティティ / プロジェクト / 名前。 |
| `source_version` | プライマリ (シーケンス) コレクション 内 の Artifact の バージョン。`v{number}` という 形式 の 文字列。 |
| `state` | Artifact の ステータス。「PENDING」、「COMMITTED」、または「DELETED」 の いずれか。 |
| `tags` | この Artifact バージョン に 割り当てられた 1 つ または 複数 の タグ の リスト。 |
| `ttl` | Artifact の Time-To-Live (TTL) ポリシー。Artifact は TTL ポリシー の 期間 が 経過 すると すぐ に 削除 されます。`None` に 設定 すると、Artifact は TTL ポリシー を 非アクティブ化 し、チーム の デフォルト TTL が あっても 削除 の スケジュール は 設定 されません。チーム 管理者 が デフォルト の TTL を 定義 し、Artifact に カスタム ポリシー が 設定 されていない 場合、Artifact は チーム の デフォルト から TTL ポリシー を 継承 します。 |
| `type` | Artifact の タイプ。一般的 な タイプ には、`dataset` や `model` があります。 |
| `updated_at` | Artifact が 最後 に 更新 された 時刻。 |
| `url` | Artifact の URL を 構築します。 |
| `version` | セカンダリ (ポートフォリオ) コレクション 内 の Artifact の バージョン。 |

## メソッド

### `add`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1487-L1578)

```python
add(
    obj: WBValue,
    name: StrPath,
    overwrite: bool = (False)
) -> ArtifactManifestEntry
```

wandb.WBValue `obj` を Artifact に 追加 します。

| arg |  |
| :--- | :--- |
| `obj` | 追加 する オブジェクト。現在、Bokeh、JoinedTable、PartitionedTable、Table、Classes、ImageMask、BoundingBoxes2D、Audio、Image、Video、Html、Object3D の いずれか 1 つ を サポート しています。 |
| `name` | オブジェクト を 追加 する Artifact 内 の パス。 |
| `overwrite` | True の 場合、同じ ファイル パス を 持つ 既存 の オブジェクト を 上書き します (該当 する 場合)。 |

| 戻り値 |  |
| :--- | :--- |
| 追加 された マニフェスト エントリー |

| 例外 |  |
| :--- | :--- |
| `ArtifactFinalizedError` | 現在 の Artifact バージョン は 確定 されているため、変更 できません。代わり に 新しい Artifact バージョン を ログ に 記録 して ください。 |

### `add_dir`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1342-L1402)

```python
add_dir(
    local_path: str,
    name: (str | None) = None,
    skip_cache: (bool | None) = (False),
    policy: (Literal['mutable', 'immutable'] | None) = "mutable"
) -> None
```

ローカル ディレクトリ を Artifact に 追加 します。

| arg |  |
| :--- | :--- |
| `local_path` | ローカル ディレクトリ の パス。 |
| `name` | Artifact 内 の サブ ディレクトリ 名。指定 する 名前 は、Artifact の `type` で ネスト された W&B App UI に 表示 されます。デフォルト は Artifact の ルート です。 |
| `skip_cache` | `True` に 設定 すると、W&B は アップロード 中 に ファイル を キャッシュ に コピー / 移動 しません。 |
| `policy` | "mutable" | "immutable"。デフォルト では、"mutable" です。"mutable" : アップロード 中 に 破損 を 防ぐ ため に、ファイル の 一時 コピー を 作成 します。"immutable" : 保護 を 無効 にし、ユーザー が ファイル を 削除 または 変更 しない こと に 依存 します。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactFinalizedError` | 現在 の Artifact バージョン は 確定 されているため、変更 できません。代わり に 新しい Artifact バージョン を ログ に 記録 して ください。 |
| `ValueError` | ポリシー は "mutable" または "immutable" で ある 必要 があります。 |

### `add_file`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1289-L1340)

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

ローカル ファイル を Artifact に 追加 します。

| arg |  |
| :--- | :--- |
| `local_path` | 追加 する ファイル の パス。 |
| `name` | 追加 する ファイル に 使用 する Artifact 内 の パス。デフォルト は ファイル の basename です。 |
| `is_tmp` | True の 場合、ファイル の 名前 が 決定 的 に 変更 され、衝突 が 回避 されます。 |
| `skip_cache` | `True` の 場合、W&B は アップロード 後 に ファイル を キャッシュ に コピー しません。 |
| `policy` | デフォルト では、"mutable" に 設定 されます。"mutable" に 設定 すると、アップロード 中 に 破損 を 防ぐ ため に、ファイル の 一時 コピー が 作成 されます。"immutable" に 設定 すると、保護 が 無効 になり、ユーザー が ファイル を 削除 または 変更 しない こと に 依存 します。 |
| `overwrite` | `True` の 場合、ファイル が 既に 存在 する 場合 は 上書き します。 |

| 戻り値 |  |
| :--- | :--- |
| 追加 された マニフェスト エントリー。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactFinalizedError` | 現在 の Artifact バージョン は 確定 されているため、変更 できません。代わり に 新しい Artifact バージョン を ログ に 記録 して ください。 |
| `ValueError` | ポリシー は "mutable" または "immutable" で ある 必要 があります。 |

### `add_reference`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1404-L1485)

```python
add_reference(
    uri: (ArtifactManifestEntry | str),
    name: (StrPath | None) = None,
    checksum: bool = (True),
    max_objects: (int | None) = None
) -> Sequence[ArtifactManifestEntry]
```

URI で 示される 参照 を Artifact に 追加 します。

Artifact に 追加 する ファイル または ディレクトリ とは 異なり、参照 は W&B に アップロード されません。詳細 については、[外部 ファイル の 追跡](https://docs.wandb.ai/guides/artifacts/track-external-files) を 参照 して ください。

デフォルト では、次 の スキーム が サポート されています。

- http(s) : ファイル の サイズ と ダイジェスト は、サーバー から 返される `Content-Length` と `ETag` レスポンス ヘッダー によって 推測 されます。
- s3 : チェックサム と サイズ は オブジェクト メタデータ から 取得 されます。バケット の バージョン管理 が 有効 になっている 場合 は、バージョン ID も 追跡 されます。
- gs : チェックサム と サイズ は オブジェクト メタデータ から 取得 されます。バケット の バージョン管理 が 有効 になっている 場合 は、バージョン ID も 追跡 されます。
- https、ドメイン が `*.blob.core.windows.net` (Azure) と 一致 する : チェックサム と サイズ は BLOB メタデータ から 取得 されます。ストレージ アカウント の バージョン管理 が 有効 になっている 場合 は、バージョン ID も 追跡 されます。
- file : チェックサム と サイズ は ファイル システム から 取得 されます。この スキーム は、追跡 する 必要 が ある が 必ずしも アップロード する 必要 が ない ファイル を 含む NFS 共有 または その他 の 外部 マウント された ボリューム が ある 場合 に 役立ちます。

その他 の スキーム の 場合、ダイジェスト は URI の ハッシュ に すぎず、サイズ は 空白 の まま に なります。

| arg |  |
| :--- | :--- |
| `uri` | 追加 する 参照 の URI パス。URI パス には、別 の Artifact の エントリー への 参照 を 格納 する ため に `Artifact.get_entry` から 返される オブジェクト を 指定 できます。 |
| `name` | この 参照 の コンテンツ を 配置 する Artifact 内 の パス。 |
| `checksum` | 参照 URI に ある リソース の チェックサム を 計算 する か どうか。チェックサム を 計算 すると、自動 整合性 検証 が 有効 になるため、強く お勧め し ます。チェックサム を 無効 にすると、Artifact の 作成 が 高速化 されますが、参照 ディレクトリ は 反復 処理 されないため、ディレクトリ 内 の オブジェクト は Artifact に 保存 されません。参照 オブジェクト を 追加 する 場合 は `checksum=False` を 設定 し、参照 URI が 変更 された 場合 にのみ 新しい バージョン が 作成 される よう に する こと を お勧め します。 |
| `max_objects` | ディレクトリ または バケット ストア プレフィックス を 指す 参照 を 追加 する 場合 に 考慮 する オブジェクト の 最大 数。デフォルト では、Amazon S3、GCS、Azure、および ローカル ファイル で 許可 される オブジェクト の 最大 数 は 10,000,000 です。その他 の URI スキーマ には 最大 値 が ありません。 |

| 戻り値 |  |
| :--- | :--- |
| 追加 された マニフェスト エントリー。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactFinalizedError` | 現在 の Artifact バージョン は 確定 されているため、変更 できません。代わり に 新しい Artifact バージョン を ログ に 記録 して ください。 |

### `checkout`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1993-L2021)

```python
checkout(
    root: (str | None) = None
) -> str
```

指定 された ルート ディレクトリ を Artifact の コンテンツ に 置き換え ます。

警告 : これ により、Artifact に 含まれていない `root` 内 の 全て の ファイル が 削除 されます。

| arg |  |
| :--- | :--- |
| `root` | この Artifact の ファイル で 置き換える ディレクトリ。 |

| 戻り値 |  |
| :--- | :--- |
| チェックアウト された コンテンツ の パス。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2131-L2150)

```python
delete(
    delete_aliases: bool = (False)
) -> None
```

Artifact と その ファイル を 削除 します。

リンクされた Artifact (つまり、ポートフォリオ コレクション の メンバー) で 呼び出された 場合 : リンク のみ が 削除 され、ソース Artifact は 影響 を 受けません。

| arg |  |
| :--- | :--- |
| `delete_aliases` | `True` に 設定 すると、Artifact に 関連付けられた 全て の エイリアス が 削除 されます。そう でない 場合、Artifact に 既存 の エイリアス が ある 場合 は 例外 が 発生 します。この パラメータ は、Artifact が リンクされている 場合 (つまり、ポートフォリオ コレクション の メンバー) は 無視 されます。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |

### `download`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1756-L1807)

```python
download(
    root: (StrPath | None) = None,
    allow_missing_references: bool = (False),
    skip_cache: (bool | None) = None,
    path_prefix: (StrPath | None) = None
) -> FilePathStr
```

Artifact の コンテンツ を 指定 された ルート ディレクトリ に ダウンロード します。

`root` 内 に ある 既存 の ファイル は 変更 されません。`root` の コンテンツ を Artifact と 正確 に 一致 させる 場合 は、`download` を 呼び出す 前 に `root` を 明示 的 に 削除 して ください。

| arg |  |
| :--- | :--- |
| `root` | W&B が Artifact の ファイル を 格納 する ディレクトリ。 |
| `allow_missing_references` | `True` に 設定 すると、参照 ファイル の ダウンロード 中 に 無効 な 参照 パス は 無視 されます。 |
| `skip_cache` | `True` に 設定 すると、ダウンロード 時 に Artifact キャッシュ が スキップ され、W&B は 各 ファイル を デフォルト の ルート または 指定 された ダウンロード ディレクトリ に ダウンロード します。 |
| `path_prefix` | 指定 すると、指定 された プレフィックス で 始まる パス を 持つ ファイル のみ が ダウンロード されます。UNIX 形式 (フォワード スラッシュ) を 使用 します。 |

| 戻り値 |  |
| :--- | :--- |
| ダウンロード された コンテンツ へ の パス。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |
| `RuntimeError` | Artifact を オフライン モード で ダウンロード しよう とした 場合。 |

### `file`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2063-L2087)

```python
file(
    root: (str | None) = None
) -> StrPath
```

単一 の ファイル Artifact を `root` で 指定 する ディレクトリ に ダウンロード します。

| arg |  |
| :--- | :--- |
| `root` | ファイル を 格納 する ルート ディレクトリ。デフォルト は './artifacts/self.name/' です。 |

| 戻り値 |  |
| :--- | :--- |
| ダウンロード された ファイル の フルパス。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |
| `ValueError` | Artifact に 複数 の ファイル が 含まれている 場合。 |

### `files`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2089-L2106)

```python
files(
    names: (list[str] | None) = None,
    per_page: int = 50
) -> ArtifactFiles
```

この Artifact に 格納 されている 全て の ファイル を 反復 処理 します。

| arg |  |
| :--- | :--- |
| `names` | リスト する Artifact の ルート に 関連 する ファイル名 パス。 |
| `per_page` | リクエスト ごと に 返す ファイル の 数。 |

| 戻り値 |  |
| :--- | :--- |
| `File` オブジェクト を 含む イテレーター。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |

### `finalize`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L902-L910)

```python
finalize() -> None
```

Artifact バージョン を 確定 します。

Artifact は 特定 の Artifact バージョン として ログ に 記録 される ため、確定 された Artifact バージョン は 変更 できません。新しい Artifact バージョン を 作成 して、Artifact に より 多く の データ を ログ に 記録 します。Artifact は、`log_artifact` で Artifact を ログ に 記録 すると 自動的 に 確定 されます。

### `get`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1673-L1718)

```python
get(
    name: str
) -> (WBValue | None)
```

Artifact 関連 の `name` に ある WBValue オブジェクト を 取得 します。

| arg |  |
| :--- | :--- |
| `name` | 取得 する Artifact 関連 の 名前。 |

| 戻り値 |  |
| :--- | :--- |
| `wandb.log()` で ログ に 記録 し、W&B UI で 可視化 できる W&B オブジェクト。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない か、run が オフライン の 場合。 |

### `get_added_local_path_name`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1720-L1732)

```python
get_added_local_path_name(
    local_path: str
) -> (str | None)
```

ローカル ファイル システム パス によって 追加 された ファイル の Artifact 関連 の 名前 を 取得 します。

| arg |  |
| :--- | :--- |
| `local_path` | Artifact 関連 の 名前 に 解決 する ローカル パス。 |

| 戻り値 |  |
| :--- | :--- |
| Artifact 関連 の 名前。 |

### `get_entry`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1652-L1671)

```python
get_entry(
    name: StrPath
) -> ArtifactManifestEntry
```

指定 された 名前 の エントリー を 取得 します。

| arg |  |
| :--- | :--- |
| `name` | 取得 する Artifact 関連 の 名前 |

| 戻り値 |  |
| :--- | :--- |
| `W&B` オブジェクト。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない か、run が オフライン の 場合。 |
| `KeyError` | Artifact に 指定 された 名前 の エントリー が 含まれていない 場合。 |

### `get_path`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1644-L1650)

```python
get_path(
    name: StrPath
) -> ArtifactManifestEntry
```

非推奨。`get_entry(name)` を 使用 して ください。

### `is_draft`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L912-L917)

```python
is_draft() -> bool
```

Artifact が 保存 されていない か どうか を 確認 します。

戻り値 : ブール値。Artifact が 保存 されている 場合 は `False`。Artifact が 保存 されていない 場合 は `True`。

### `json_encode`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2338-L2345)

```python
json_encode() -> dict[str, Any]
```

Artifact を JSON 形式 に エンコード して 返します。

| 戻り値 |  |
| :--- | :--- |
| Artifact の 属性 を 表す `string` キー を 持つ `dict`。 |

### `link`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2177-L2205)

```python
link(
    target_path: str,
    aliases: (list[str] | None) = None
) -> None
```

この Artifact を ポートフォリオ (Artifact の 昇格 された コレクション) に リンク します。

| arg |  |
| :--- | :--- |
| `target_path` | プロジェクト 内 の ポートフォリオ へ の パス。ターゲット パス は、`{portfolio}`、`{project}/{portfolio}`、または `{entity}/{project}/{portfolio}` の いずれか の スキーマ に 準拠 して いる 必要 があります。Artifact を プロジェクト 内 の 一般 的 な ポートフォリオ で はなく、Model Registry に リンク する には、`target_path` を 次 の スキーマ `{model-registry}/{Registered Model Name}` または `{entity}/{model-registry}/{Registered Model Name}` に 設定 します。 |
| `aliases` | 指定 された ポートフォリオ 内 の Artifact を 一意 に 識別 する 文字列 の リスト。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |

### `logged_by`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2294-L2336)

```python
logged_by() -> (Run | None)
```

Artifact を 最初 に ログ に 記録 した W&B run を 取得 します。

| 戻り値 |  |
| :--- | :--- |
| Artifact を 最初 に ログ に 記録 した W&B run の 名前。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |

### `new_draft`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L424-L457)

```python
new_draft() -> Artifact
```

この コミット された Artifact と 同じ コンテンツ を 持つ 新しい ドラフト Artifact を 作成 します。

既存 の Artifact を 変更 すると、「増分 Artifact」 と 呼ば れる 新しい Artifact バージョン が 作成 されます。返された Artifact は、拡張 または 変更 して 新しい バージョン として ログ に 記録 できます。

| 戻り値 |  |
| :--- | :--- |
| `Artifact` オブジェクト。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |

### `new_file`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1246-L1287)

```python
@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "x",
    encoding: (str | None) = None
) -> Iterator[IO]
```

新しい 一時 ファイル を 開き、Artifact に 追加 します。

| arg |  |
| :--- | :--- |
| `name` | Artifact に 追加 する 新しい ファイル の 名前。 |
| `mode` | 新しい ファイル を 開く ため に 使用 する ファイル アクセス モード。 |
| `encoding` | 新しい ファイル を 開く ため に 使用 する エンコード。 |

| 戻り値 |  |
| :--- | :--- |
| 書き込み 可能 な 新しい ファイル オブジェクト。閉じると、ファイル は 自動的 に Artifact に 追加 されます。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactFinalizedError` | 現在 の Artifact バージョン は 確定 されているため、変更 できません。代わり に 新しい Artifact バージョン を ログ に 記録 して ください。 |

### `remove`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1614-L1642)

```python
remove(
    item: (StrPath | ArtifactManifestEntry)
) -> None
```

Artifact から アイテム を 削除 します。

| arg |  |
| :--- | :--- |
| `item` | 削除 する アイテム。特定 の マニフェスト エントリー または Artifact 関連 の パス の 名前 に する ことができます。アイテム が ディレクトリ と 一致 する 場合、その ディレクトリ 内 の 全て の アイテム が 削除 されます。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactFinalizedError` | 現在 の Artifact バージョン は 確定 されているため、変更 できません。代わり に 新しい Artifact バージョン を ログ に 記録 して ください。 |
| `FileNotFoundError` | Artifact に アイテム が 見つからない 場合。 |

### `save`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L922-L961)

```python
save(
    project: (str | None) = None,
    settings: (wandb.Settings | None) = None
) -> None
```

Artifact に 行われた 変更 を 永続 化 します。

現在 run 内 に ある 場合、その run は この Artifact を ログ に 記録 します。現在 run 内 に ない 場合、「auto」 タイプ の run が 作成 され、この Artifact を 追跡 します。

| arg |  |
| :--- | :--- |
| `project` | run が 既に コンテキスト 内 に ない 場合 に Artifact に 使用 する プロジェクト。 |
| `settings` | 自動 run を 初期化 する とき に 使用 する settings オブジェクト。最も 一般 的 に テスト ハーネス で 使用 されます。 |

### `unlink`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2207-L2222)

```python
unlink() -> None
```

この Artifact が 現在 ポートフォリオ (Artifact の 昇格 された コレクション) の メンバー で ある 場合 は、リンク を 解除 します。

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |
| `ValueError` | Artifact が リンクされていない 場合、つまり ポートフォリオ コレクション の メンバー でない 場合。 |

### `used_by`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2248-L2292)

```python
used_by() -> list[Run]
```

この Artifact を 使用 した run の リスト を 取得 します。

| 戻り値 |  |
| :--- | :--- |
| `Run` オブジェクト の リスト。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |

### `verify`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L2023-L2061)

```python
verify(
    root: (str | None) = None
) -> None
```

Artifact の コンテンツ が マニフェスト と 一致 する こと を 確認 します。

ディレクトリ 内 の 全て の ファイル が チェックサム され、チェックサム が Artifact の マニフェスト と 相互 参照 されます。参照 は 検証 されません。

| arg |  |
| :--- | :--- |
| `root` | 検証 する ディレクトリ。None の 場合、Artifact は './artifacts/self.name/' に ダウンロード されます。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない 場合。 |
| `ValueError` | 検証 に 失敗 した場合。 |

### `wait`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L971-L995)

```python
wait(
    timeout: (int | None) = None
) -> Artifact
```

必要 に 応じ て、この Artifact が ログ 記録 を 完了 する まで 待ちます。

| arg |  |
| :--- | :--- |
| `timeout` | 待機 する 時間 (秒単位)。 |

| 戻り値 |  |
| :--- | :--- |
| `Artifact` オブジェクト。 |

### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1216-L1228)

```python
__getitem__(
    name: str
) -> (WBValue | None)
```

Artifact 関連 の `name` に ある WBValue オブジェクト を 取得 します。

| arg |  |
| :--- | :--- |
| `name` | 取得 する Artifact 関連 の 名前。 |

| 戻り値 |  |
| :--- | :--- |
| `wandb.log()` で ログ に 記録 し、W&B UI で 可視化 できる W&B オブジェクト。 |

| 例外 |  |
| :--- | :--- |
| `ArtifactNotLoggedError` | Artifact が ログ に 記録 されていない か、run が オフライン の 場合。 |

### `__setitem__`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/artifacts/artifact.py#L1230-L1244)

```python
__setitem__(
    name: str,
    item: WBValue
) -> ArtifactManifestEntry
```

パス `name` に ある Artifact に `item` を 追加 します。

| arg |  |
| :--- | :--- |
| `name` | オブジェクト を 追加 する Artifact 内 の パス。 |
| `item` | 追加 する オブジェクト。 |

| 戻り値 |  |
| :--- | :--- |
| 追加 された マニフェスト エントリー |

| 例外 |  |
| :--- | :--- |
| `ArtifactFinalizedError` | 現在 の Artifact バージョン は 確定 されているため、変更 できません。代わり に 新しい Artifact バージョン を ログ に 記録 して ください。 |
