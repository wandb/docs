---
title: アーティファクトを削除する
description: App UI を使って対話的に、または W&B SDK を使ってプログラムからアーティファクトを削除できます。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-delete-artifacts
    parent: manage-data
---

App UI を使って対話的に、または W&B SDK を使ってプログラムから Artifacts を削除できます。Artifact を削除すると、W&B はその artifact を *ソフトデリート* 状態にします。つまり、artifact が削除予定としてマークされますが、ファイル自体はすぐにはストレージから削除されません。

artifact の内容は、ソフトデリートまたは削除保留状態のまま保持され、定期的に実行されるガベージコレクション プロセスが、削除予定としてマークされた全ての artifact を確認します。artifact や関連ファイルが以前や以降の artifact バージョンで使われていない場合、このガベージコレクションプロセスにより関連ファイルがストレージから削除されます。

このページの各セクションでは、特定の artifact バージョンの削除方法、artifact コレクションの削除方法、エイリアスがある場合・ない場合の artifact 削除方法などについて説明しています。artifact の削除タイミングは TTL ポリシーでスケジュールできます。詳しくは [Artifact TTL ポリシーによるデータ保持管理]({{< relref path="./ttl.md" lang="ja" >}}) をご覧ください。

{{% alert %}}
TTL ポリシーで削除予定にされた Artifact、W&B SDK または W&B App UI で削除した Artifact は、まずソフトデリートされます。ソフトデリート状態になった Artifact は、完全削除（ハードデリート）前にガベージコレクションが実施されます。
{{% /alert %}}

### アーティファクトバージョンの削除

artifact バージョンを削除するには:

1. 削除したい artifact の名前を選択します。artifact ビューが展開され、その artifact に関連付けられた全バージョンが表示されます。
2. artifact 一覧から削除したい artifact バージョンを選びます。
3. workspace 右側のケバブドロップダウンを選択します。
4. 「Delete」を選択します。

artifact バージョンは [delete()]({{< relref path="/ref/python/sdk/classes/artifact.md#delete" lang="ja" >}}) メソッドを使ってプログラムからも削除できます。下記の例を参照してください。

### エイリアス付きで複数の artifact バージョンを削除する

下記のコード例は、エイリアスが付与されている artifacts をまとめて削除する方法を示しています。entity、project 名、artifacts を作成した run ID を指定してください。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

artifact に 1つ以上のエイリアスがある場合、それも削除したい場合は `delete_aliases` パラメータに `True` を設定してください。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # エイリアスごと削除したい場合は delete_aliases=True を指定
    artifact.delete(delete_aliases=True)
```

### 特定のエイリアスを持つ複数 artifact バージョンの削除

以下のコードは、特定エイリアスを持つ複数の artifact バージョンを削除する方法の一例です。entity、project 名、artifacts を作成した run ID を指定し、自分用の削除ロジックに置き換えてください。

```python
import wandb

runs = api.run("entity/project_name/run_id")

# 'v3' および 'v4' というエイリアス付きartifactを削除
for artifact_version in runs.logged_artifacts():
    # ここを自身の削除ロジックに置き換えてください。
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### エイリアスがない artifact の全バージョンを削除

以下のコードスニペットは、エイリアスがない artifact の全バージョンを削除する方法を示します。`wandb.Api` の `project` および `entity` キーにプロジェクト名とエンティティ名を入れてください。`<>` は自身の artifact 名に置き換えてください。

```python
import wandb

# wandb.Api メソッド使用時は entity と project 名を指定
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # タイプと名前を指定
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest' などのエイリアスが付いていないバージョンを削除
    # ※任意の削除ロジックを書き換えて構いません
    if len(v.aliases) == 0:
        v.delete()
```

### artifact コレクションを削除する

artifact コレクションを削除するには:

1. 削除したい artifact コレクションに移動し、マウスオーバーします。
2. artifact コレクション名の隣にあるケバブドロップダウンを選択します。
3. 「Delete」を選択します。

artifact コレクションも [delete()]({{< relref path="/ref/python/sdk/classes/artifact.md#delete" lang="ja" >}}) メソッドでプログラムから削除できます。`wandb.Api` の `project` と `entity` キーにプロジェクト名・エンティティ名を指定してください。

```python
import wandb

# wandb.Api メソッド使用時は entity と project 名を指定
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&B のホスティング形態別ガベージコレクション有効化方法

W&B の共有クラウドをご利用の場合、ガベージコレクションはデフォルトで有効化されています。W&B のホスティング方法によっては追加設定が必要になる場合があります。例:

* `GORILLA_ARTIFACT_GC_ENABLED` 環境変数を true に設定: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/object-versioning)、または [Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning) のようなストレージプロバイダをご利用の場合はバケットのバージョン管理を有効化してください。Azure をご利用の場合は [ソフト削除の有効化](https://learn.microsoft.com/azure/storage/blobs/soft-delete-blob-overview) を行ってください。

  {{% alert %}}
  Azure のソフト削除は、他のストレージプロバイダにおけるバケットのバージョン管理と同等です。
  {{% /alert %}}

以下の表は、デプロイ形態ごとにガベージコレクションを有効化するための要件をまとめています。

「X」は該当要件の対応が必要であることを示します。

|                                                | 環境変数                | バージョン管理有効化 | 
| -----------------------------------------------| ------------------------| ------------------ | 
| 共有クラウド                                   |                         |                    | 
| 共有クラウド + [セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) |                         | X                  | 
| 専用クラウド                                   |                         |                    | 
| 専用クラウド + [セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) |                         | X                  | 
| カスタマー管理クラウド                         | X                       | X                  | 
| カスタマー管理オンプレミス                     | X                       | X                  |

{{% alert %}}note
セキュアストレージコネクタは現時点では Google Cloud Platform および Amazon Web Services でのみ利用可能です。
{{% /alert %}}