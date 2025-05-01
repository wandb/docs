---
title: アーティファクトを削除する
description: アプリ UI を使って対話的に、または W&B SDK を使用してプログラムでアーティファクトを削除します。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-delete-artifacts
    parent: manage-data
---

アーティファクトは、App UI でインタラクティブに削除するか、W&B SDK でプログラム的に削除できます。アーティファクトを削除すると、W&B はそのアーティファクトを*ソフト削除*としてマークします。つまり、そのアーティファクトは削除予定としてマークされますが、ファイルはすぐにはストレージから削除されません。

アーティファクトの内容は、定期的に実行されるガベージコレクション プロセスが削除対象としてマークされたすべてのアーティファクトをレビューするまで、ソフト削除または削除保留状態のままです。ガベージコレクションプロセスは、アーティファクトおよびその関連ファイルが以前または後のアーティファクトバージョンで使用されていない場合、関連ファイルをストレージから削除します。

このページのセクションでは、特定のアーティファクトバージョンの削除方法、アーティファクトコレクションの削除方法、エイリアスを持つアーティファクトの削除方法、エイリアスがないアーティファクトの削除方法などについて説明します。アーティファクトが W&B から削除される時間を TTL ポリシーでスケジュールできます。詳細については、[Artifact TTL ポリシーによるデータ保持の管理]({{< relref path="./ttl.md" lang="ja" >}})を参照してください。

{{% alert %}}
TTL ポリシーで削除予定のアーティファクト、W&B SDK で削除されたアーティファクト、または W&B App UI で削除されたアーティファクトは、最初にソフト削除されます。ソフト削除されたアーティファクトは、最終的にハード削除される前にガベージコレクションを受けます。
{{% /alert %}}

### アーティファクトバージョンの削除

アーティファクトバージョンを削除するには:

1. アーティファクトの名前を選択します。これにより、アーティファクトビューが拡張され、そのアーティファクトに関連付けられたすべてのアーティファクトバージョンが一覧表示されます。
2. アーティファクトのリストから、削除したいアーティファクトバージョンを選択します。
3. ワークスペースの右側にあるケバブドロップダウンを選択します。
4. 「Delete」を選択します。

アーティファクトバージョンは、[delete()]({{< relref path="/ref/python/artifact#delete" lang="ja" >}}) メソッドを使用してプログラム的にも削除できます。以下の例を参照してください。

### エイリアスを持つ複数のアーティファクトバージョンの削除

次のコード例は、エイリアスを持つアーティファクトを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、および run ID を指定してください。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

アーティファクトに 1 つ以上のエイリアスがある場合、`delete_aliases` パラメータをブール値 `True` に設定してエイリアスを削除します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # delete_aliases=True を設定して、
    # エイリアスを 1 つ以上持つアーティファクトを削除します
    artifact.delete(delete_aliases=True)
```

### 特定のエイリアスを持つ複数のアーティファクトバージョンを削除

以下のコードは、特定のエイリアスを持つ複数のアーティファクトバージョンを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、および run ID を指定します。削除ロジックは独自のものに置き換えてください:

```python
import wandb

runs = api.run("entity/project_name/run_id")

# 'v3' および 'v4' のエイリアスを持つアーティファクトを削除
for artifact_version in runs.logged_artifacts():
    # 独自の削除ロジックに置き換えます。
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### エイリアスを持たないアーティファクトのすべてのバージョンを削除

次のコードスニペットは、エイリアスを持たないアーティファクトのすべてのバージョンを削除する方法を示しています。`wandb.Api` の `project` および `entity` キーにプロジェクト名とエンティティの名前をそれぞれ指定してください。`<>` をアーティファクトの名前に置き換えてください:

```python
import wandb

# wandb.Api メソッドを使用する際に、エンティティとプロジェクト名を指定してください。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # タイプと名前を指定
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest' などのエイリアスを持たないバージョンをクリーンアップします。
    # 注意: ここには任意の削除ロジックを置くことができます。
    if len(v.aliases) == 0:
        v.delete()
```

### アーティファクトコレクションの削除

アーティファクトコレクションを削除するには:

1. 削除したいアーティファクトコレクションに移動し、その上にカーソルを合わせます。
2. アーティファクトコレクション名の横にあるケバブドロップダウンを選択します。
4. 「Delete」を選択します。

アーティファクトコレクションは、[delete()]({{< relref path="/ref/python/artifact.md#delete" lang="ja" >}}) メソッドを使用してプログラムで削除することもできます。`wandb.Api` の `project` および `entity` キーにプロジェクト名とエンティティの名前を指定してください:

```python
import wandb

# wandb.Api メソッドを使用する際に、エンティティとプロジェクト名を指定してください。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&B がホストされている方法に基づいたガベージコレクションの有効化方法

W&B の共有クラウドを使用している場合、ガベージコレクションはデフォルトで有効です。W&B をホストする方法に基づいて、ガベージコレクションを有効にするために追加の手順が必要な場合があります。これには以下が含まれます:

* `GORILLA_ARTIFACT_GC_ENABLED` 環境変数を true に設定: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/object-versioning) または [Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning) などのストレージプロバイダーを使用している場合は、バケットバージョン管理を有効にします。Azure を使用している場合は、[ソフト削除を有効](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview)にします。
  {{% alert %}}
  Azure のソフト削除は、他のストレージプロバイダーでのバケットバージョン管理に相当します。
  {{% /alert %}}

以下の表は、デプロイメントタイプに基づいてガベージコレクションを有効にするための要件を満たす方法を説明しています。

`X` は要件を満たす必要があることを示します:

|                                                | 環境変数                 | バージョン管理の有効化 | 
| -----------------------------------------------| ------------------------| ----------------- | 
| 共有クラウド                                   |                         |                   | 
| [セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を使用した共有クラウド |                         | X                 | 
| 専用クラウド                                |                         |                   | 
| [セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を使用した専用クラウド |                         | X                 | 
| カスタマーマネージドクラウド                         | X                       | X                 | 
| カスタマーマネージドオンプレミス                       | X                       | X                 |

{{% alert %}}注意
セキュアストレージコネクタは現在、Google Cloud Platform および Amazon Web Services のみで利用可能です。
{{% /alert %}}