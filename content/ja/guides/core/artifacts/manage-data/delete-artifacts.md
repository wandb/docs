---
title: Delete an artifact
description: App UI を使用してインタラクティブに、または W&B SDK を使用してプログラム的に Artifacts を削除します。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-delete-artifacts
    parent: manage-data
---

App UIまたは W&B SDK を使用して、アーティファクトをインタラクティブに、またはプログラムで削除できます。アーティファクトを削除すると、W&B はそのアーティファクトを *論理削除* としてマークします。つまり、アーティファクトは削除対象としてマークされますが、ファイルはストレージからすぐに削除されるわけではありません。

アーティファクトの内容は、論理削除、つまり削除保留状態のままとなり、定期的に実行されるガベージコレクション プロセスが、削除対象としてマークされたすべてのアーティファクトを確認します。ガベージコレクション プロセスは、アーティファクトとその関連ファイルが、以前または以降のアーティファクト バージョンで使用されていない場合、ストレージから関連ファイルを削除します。

このページのセクションでは、特定のアーティファクト バージョンを削除する方法、アーティファクト コレクションを削除する方法、エイリアスを使用または使用せずにアーティファクトを削除する方法などについて説明します。アーティファクトが W&B から削除されるタイミングは、TTL ポリシーでスケジュールできます。詳細については、[アーティファクト TTL ポリシーによるデータ保持の管理]({{< relref path="./ttl.md" lang="ja" >}}) を参照してください。

{{% alert %}}
TTL ポリシーで削除がスケジュールされているアーティファクト、W&B SDK で削除されたアーティファクト、または W&B App UI で削除されたアーティファクトは、最初に論理削除されます。論理削除されたアーティファクトは、完全に削除される前にガベージコレクションの対象となります。
{{% /alert %}}

### アーティファクト バージョンを削除する

アーティファクト バージョンを削除するには:

1. アーティファクトの名前を選択します。これにより、アーティファクト ビューが展開され、そのアーティファクトに関連付けられているすべてのアーティファクト バージョンが一覧表示されます。
2. アーティファクトのリストから、削除するアーティファクト バージョンを選択します。
3. ワークスペースの右側にあるケバブ ドロップダウンを選択します。
4. 削除を選択します。

アーティファクト バージョンは、[delete()]({{< relref path="/ref/python/artifact#delete" lang="ja" >}}) メソッドを使用してプログラムで削除することもできます。以下の例を参照してください。

### エイリアスを持つ複数のアーティファクト バージョンを削除する

次のコード例は、エイリアスが関連付けられているアーティファクトを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、run ID を指定します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

アーティファクトに1つ以上のエイリアスがある場合にエイリアスを削除するには、`delete_aliases` パラメータをブール値 `True` に設定します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # 1つ以上エイリアスを持つ
    # アーティファクトを削除するには
    # delete_aliases=True を設定します
    artifact.delete(delete_aliases=True)
```

### 特定のエイリアスを持つ複数のアーティファクト バージョンを削除する

次のコードは、特定のエイリアスを持つ複数のアーティファクト バージョンを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、run ID を指定します。削除ロジックを独自のものに置き換えます。

```python
import wandb

runs = api.run("entity/project_name/run_id")

# エイリアス 'v3' と 'v4' を持つアーティファクトを削除します
for artifact_version in runs.logged_artifacts():
    # 独自の削除ロジックに置き換えます。
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### エイリアスを持たないアーティファクトのすべてのバージョンを削除する

次のコードスニペットは、エイリアスを持たないアーティファクトのすべてのバージョンを削除する方法を示しています。`wandb.Api` の `project` および `entity` キーに、プロジェクトとエンティティの名前をそれぞれ指定します。`<>` をアーティファクトの名前に置き換えます。

```python
import wandb

# wandb.Api メソッドを使用する場合は、
# エンティティとプロジェクト名を指定してください。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # タイプと名前を指定
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest' などのエイリアスを持たないバージョンをクリーンアップします。
    # 注: ここには、必要な削除ロジックを入れることができます。
    if len(v.aliases) == 0:
        v.delete()
```

### アーティファクト コレクションを削除する

アーティファクト コレクションを削除するには:

1. 削除するアーティファクト コレクションに移動し、その上にカーソルを置きます。
3. アーティファクト コレクション名の横にあるケバブ ドロップダウンを選択します。
4. 削除を選択します。

[delete()]({{< relref path="/ref/python/artifact.md#delete" lang="ja" >}}) メソッドを使用して、プログラムでアーティファクト コレクションを削除することもできます。`wandb.Api` の `project` および `entity` キーに、プロジェクトとエンティティの名前をそれぞれ指定します。

```python
import wandb

# wandb.Api メソッドを使用する場合は、
# エンティティとプロジェクト名を指定してください。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&B のホスト方法に基づいてガベージコレクションを有効にする方法
W&B の共有クラウドを使用している場合、ガベージコレクションはデフォルトで有効になっています。W&B のホスト方法によっては、ガベージコレクションを有効にするために追加の手順が必要になる場合があります。これには以下が含まれます。

* `GORILLA_ARTIFACT_GC_ENABLED` 環境変数を true に設定します: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/object-versioning) 、または [Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning) などの他のストレージ プロバイダーを使用する場合は、バケットの バージョン管理を有効にします。Azure を使用する場合は、[論理削除を有効にします](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview)。
  {{% alert %}}
  Azure の論理削除は、他のストレージ プロバイダーのバケットの バージョン管理と同等です。
  {{% /alert %}}

次の表は、デプロイメントの種類に基づいてガベージコレクションを有効にするための要件を満たす方法を示しています。

`X` は、要件を満たす必要があることを示します。

|                                                | 環境変数    | バージョン管理を有効にする |
| -----------------------------------------------| ------------------------| ----------------- |
| 共有クラウド                                   |                         |                   |
| [セキュア ストレージ コネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を使用した共有クラウド|                         | X                 |
| 専用クラウド                                |                         |                   |
| [セキュア ストレージ コネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を使用した専用クラウド|                         | X                 |
| 顧客管理クラウド                         | X                       | X                 |
| 顧客管理オンプレミス                       | X                       | X                 |

{{% alert %}}note
セキュア ストレージ コネクタは、現在 Google Cloud Platform および Amazon Web Services でのみ利用可能です。
{{% /alert %}}
