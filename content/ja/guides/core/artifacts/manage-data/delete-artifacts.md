---
title: Delete an artifact
description: App UI を使用してインタラクティブに、または W&B SDK を使用してプログラムで Artifacts を削除します。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-delete-artifacts
    parent: manage-data
---

App UIまたは W&B SDKを使用して、アーティファクトをインタラクティブに、またはプログラムで削除します。アーティファクトを削除すると、W&Bはそのアーティファクトを _ソフトデリート_ としてマークします。言い換えると、アーティファクトは削除対象としてマークされますが、ファイルはストレージからすぐに削除されません。

アーティファクトの内容は、ソフトデリート、つまり削除保留状態のままとなり、定期的に実行されるガベージコレクションプロセスが削除対象としてマークされたすべてのアーティファクトを確認します。ガベージコレクションプロセスは、アーティファクトとそれに関連するファイルが以前または以降のアーティファクトのバージョンで使用されていない場合、ストレージから関連ファイルを削除します。

このページのセクションでは、特定のアーティファクトバージョンを削除する方法、アーティファクトコレクションを削除する方法、エイリアスのあるアーティファクトとエイリアスのないアーティファクトを削除する方法などについて説明します。TTLポリシーを使用して、アーティファクトが W&Bから削除されるタイミングをスケジュールできます。詳細については、[Artifact TTLポリシーによるデータ保持の管理]({{< relref path="./ttl.md" lang="ja" >}})を参照してください。

{{% alert %}}
TTLポリシーで削除がスケジュールされている Artifacts、W&B SDKで削除された Artifacts、または W&B App UIで削除された Artifactsは、最初にソフトデリートされます。ソフトデリートされた Artifacts は、ハードデリートされる前にガベージコレクションを受けます。
{{% /alert %}}

### アーティファクト バージョンの削除

アーティファクト バージョンを削除するには:

1. アーティファクトの名前を選択します。これにより、アーティファクト ビューが展開され、そのアーティファクトに関連付けられているすべてのアーティファクト バージョンが一覧表示されます。
2. アーティファクトのリストから、削除するアーティファクト バージョンを選択します。
3. ワークスペースの右側にあるケバブドロップダウンを選択します。
4. [削除] を選択します。

アーティファクト バージョンは、[delete()]({{< relref path="/ref/python/artifact#delete" lang="ja" >}}) メソッドを使用してプログラムで削除することもできます。以下の例を参照してください。

### エイリアスを持つ複数のアーティファクト バージョンを削除する

次のコード例は、エイリアスが関連付けられているアーティファクトを削除する方法を示しています。アーティファクトを作成した entity、project 名、および run IDを指定します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

`delete_aliases` パラメータをブール値 `True` に設定して、アーティファクトに1つ以上のエイリアスがある場合にエイリアスを削除します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # 1つ以上エイリアスを持つ artifacts を削除するために、delete_aliases=True に設定します。
    artifact.delete(delete_aliases=True)
```

### 特定のエイリアスを持つ複数のアーティファクト バージョンを削除する

次のコードは、特定のエイリアスを持つ複数のアーティファクト バージョンを削除する方法を示しています。アーティファクトを作成した entity、project 名、および run IDを指定します。削除ロジックを独自のものに置き換えます。

```python
import wandb

runs = api.run("entity/project_name/run_id")

# アーティファクト ith alias 'v3' および 'v4' を削除
for artifact_version in runs.logged_artifacts():
    # 独自の削除ロジックに置き換えます。
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### エイリアスを持たないアーティファクトのすべてのバージョンを削除する

次のコードスニペットは、エイリアスを持たないアーティファクトのすべてのバージョンを削除する方法を示しています。`wandb.Api` の `project` および `entity` キーに、project および entity の名前をそれぞれ指定します。`<>` をアーティファクトの名前に置き換えます。

```python
import wandb

# wandb.Api メソッドを使用する場合は、entity と project 名を指定してください。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # type と名前を指定
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest' などのエイリアスを持たないバージョンをクリーンアップします。
    # 注: ここに任意の削除ロジックを記述できます。
    if len(v.aliases) == 0:
        v.delete()
```

### アーティファクト コレクションを削除する

アーティファクト コレクションを削除するには:

1. 削除するアーティファクト コレクションに移動し、その上にカーソルを置きます。
2. アーティファクト コレクション名の横にあるケバブドロップダウンを選択します。
3. [削除] を選択します。

[delete()]({{< relref path="/ref/python/artifact.md#delete" lang="ja" >}}) メソッドを使用して、プログラムでアーティファクト コレクションを削除することもできます。`wandb.Api` の `project` および `entity` キーに、project および entity の名前をそれぞれ指定します。

```python
import wandb

# wandb.Api メソッドを使用する場合は、entity と project 名を指定してください。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&Bのホスト方法に基づいてガベージコレクションを有効にする方法

W&B の共有クラウドを使用している場合、ガベージコレクションはデフォルトで有効になっています。W&B のホスト方法によっては、ガベージコレクションを有効にするために追加の手順が必要になる場合があります。これには以下が含まれます。

* `GORILLA_ARTIFACT_GC_ENABLED` 環境変数をtrueに設定します: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/object-versioning)または[Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning)などの他のストレージプロバイダーを使用する場合は、バケットのバージョニングを有効にします。Azureを使用する場合は、[ソフト削除を有効にします](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview)。
  {{% alert %}}
  Azureのソフト削除は、他のストレージプロバイダーのバケットのバージョニングと同等です。
  {{% /alert %}}

次の表は、デプロイメントタイプに基づいてガベージコレクションを有効にするための要件を満たす方法について説明しています。

`X` は、要件を満たす必要があることを示します。

|                                                | 環境変数    | バージョニングの有効化 | 
| -----------------------------------------------| ------------------------| ----------------- | 
| 共有クラウド                                   |                         |                   | 
| [セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を使用した共有クラウド|                         | X                 | 
| 専用クラウド                                |                         |                   | 
| [セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を使用した専用クラウド|                         | X                 | 
| お客様管理のクラウド                         | X                       | X                 | 
| お客様管理のオンプレミス                       | X                       | X                 |
 

{{% alert %}}note
セキュアストレージコネクタは、現在 Google Cloud Platform と Amazon Web Services でのみ利用可能です。
{{% /alert %}}
