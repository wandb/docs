---
description: アプリUIを使用してインタラクティブに、またはW&B SDKを使用してプログラムでArtifactsを削除します。
displayed_sidebar: default
---


# Delete artifacts

<head>
  <title>Delete W&B Artifacts</title>
</head>

アーティファクトを削除するには、App UIを使って対話的に行うか、W&B SDKを使ってプログラムで行うことができます。アーティファクトを削除すると、W&Bはそのアーティファクトを*ソフトデリート*としてマークします。つまり、アーティファクトは削除対象としてマークされますが、ファイルはすぐにはストレージから削除されません。

アーティファクトの内容は、ソフトデリートまたは保留中の削除状態のままとなり、定期的に実行されるガベージコレクションプロセスが削除対象としてマークされたすべてのアーティファクトをレビューするまで続きます。ガベージコレクションプロセスは、アーティファクトおよびその関連ファイルが前または後のアーティファクトバージョンによって使用されていない場合、関連ファイルをストレージから削除します。

このページのセクションでは、特定のアーティファクトバージョンの削除方法、アーティファクトコレクションの削除方法、エイリアスの有無にかかわらずアーティファクトを削除する方法などについて説明します。TTLポリシーを使用して、アーティファクトがW&Bから削除されるタイミングをスケジュールできます。詳細については、[Manage data retention with Artifact TTL policy](./ttl.md)を参照してください。

:::note
TTLポリシーで削除がスケジュールされたアーティファクト、W&B SDKで削除されたアーティファクト、またはW&B App UIで削除されたアーティファクトは最初にソフトデリートされます。ソフトデリートされたアーティファクトは、ハードデリートされる前にガベージコレクションの処理を受けます。
:::

### アーティファクトバージョンを削除する

アーティファクトバージョンを削除するには:

1. アーティファクトの名前を選択します。これによりアーティファクトビューが展開され、そのアーティファクトに関連付けられているすべてのアーティファクトバージョンが表示されます。
2. アーティファクトのリストから、削除したいアーティファクトバージョンを選択します。
3. ワークスペースの右側にあるケバブドロップダウンを選択します。
4. 削除を選択します。

アーティファクトバージョンは、[delete()](https://docs.wandb.ai/ref/python/artifact#delete)メソッドを使用してプログラムからも削除できます。以下の例を参照してください。

### エイリアスを持つ複数のアーティファクトバージョンを削除する

次のコード例は、エイリアスが関連付けられているアーティファクトを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、およびrun IDを指定します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

`delete_aliases`パラメータをブール型の値`True`に設定すると、エイリアスを持つアーティファクトが削除されます。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # エイリアスを持つアーティファクトを削除するには
    # delete_aliases=Trueを設定してください
    artifact.delete(delete_aliases=True)
```

### 特定のエイリアスを持つ複数のアーティファクトバージョンを削除する

以下のコードは、特定のエイリアスを持つ複数のアーティファクトバージョンを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、およびrun IDを指定します。削除ロジックを自分のものに置き換えてください:

```python
import wandb

runs = api.run("entity/project_name/run_id")

# エイリアス'v3'と'v4'を持つアーティファクトを削除します
for artifact_version in runs.logged_artifacts():
    # 自分の削除ロジックに置き換えてください。
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### エイリアスを持たないすべてのアーティファクトバージョンを削除する

次のコードスニペットは、エイリアスを持たないすべてのアーティファクトバージョンを削除する方法を示しています。プロジェクトおよびエンティティの名前を`project`および`entity`キーにそれぞれ指定します。`<>`をあなたのアーティファクトの名前に置き換えてください：

```python
import wandb

# wandb.Apiメソッドを使用する際に、エンティティとプロジェクト名を指定します。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # タイプと名前を指定
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest'などのエイリアスを持たないバージョンをクリーンアップします。
    # 注意: ここに任意の削除ロジックを入れることができます。
    if len(v.aliases) == 0:
        v.delete()
```

### アーティファクトコレクションを削除する

アーティファクトコレクションを削除するには:

1. 削除したいアーティファクトコレクションに移動し、その上にカーソルを置きます。
2. アーティファクトコレクション名の横にあるケバブドロップダウンを選択します。
3. 削除を選択します。

アーティファクトコレクションは、[delete()](../../ref/python/artifact.md#delete)メソッドを使用してプログラムから削除することもできます。プロジェクトおよびエンティティの名前を`project`および`entity`キーにそれぞれ指定します：

```python
import wandb

# wandb.Apiメソッドを使用する際に、エンティティとプロジェクト名を指定します。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&Bのホスティング方法に基づいてガベージコレクションを有効にする方法
W&Bの共有クラウドを使用している場合、ガベージコレクションはデフォルトで有効になっています。W&Bのホスティング方法に基づいて、ガベージコレクションを有効にするために追加の手順が必要になる場合があります。これには以下が含まれます：

* `GORILLA_ARTIFACT_GC_ENABLED`環境変数をtrueに設定する: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/object-versioning)または[Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning)などの他のストレージプロバイダーを使用する場合、バケットバージョン管理を有効にする。Azureを使用する場合は、[ソフトデリート](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview)を有効にします。
  :::note
  Azureのソフトデリートは、他のストレージプロバイダーにおけるバケットバージョン管理と同等です。
  :::

次の表は、デプロイメントタイプに基づいてガベージコレクションを有効にするための要件を満たす方法を説明しています。

`X`は要件を満たす必要があることを示します：

|                                                | 環境変数                | バージョン管理の有効化 | 
| -----------------------------------------------| ------------------------| -------------------- | 
| 共有クラウド                                   |                         |                      | 
| [secure storage connector](../hosting/data-security/secure-storage-connector.md) を使った共有クラウド|                         | X                    | 
| 専用クラウド                                   |                         |                      | 
| [secure storage connector](../hosting/data-security/secure-storage-connector.md) を使った専用クラウド|                         | X                    | 
| 顧客管理のクラウド                             | X                       | X                    | 
| 顧客管理のオンプレ                             | X                       | X                    | 


:::note
Secure storage connectorは現在、Google Cloud PlatformとAmazon Web Servicesのみで利用可能です。
:::