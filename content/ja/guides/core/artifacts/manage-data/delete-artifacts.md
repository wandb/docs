---
title: Delete an artifact
description: App UI を使って対話的に、または W&B SDK でプログラム的にアーティファクトを削除します。
menu:
  default:
    identifier: ja-guides-core-artifacts-manage-data-delete-artifacts
    parent: manage-data
---

アーティファクトは、App UIを使ってインタラクティブに、またはW&B SDKを使ってプログラムで削除できます。アーティファクトを削除すると、W&Bはそのアーティファクトを*ソフトデリート*としてマークします。つまり、アーティファクトは削除対象としてマークされますが、ファイルはすぐにはストレージから削除されません。

アーティファクトの内容は、ソフトデリート、または削除保留状態として残り、定期的に実行されるガベージコレクションプロセスが削除対象としてマークされたすべてのアーティファクトを確認します。ガベージコレクションプロセスは、アーティファクトとその関連ファイルが前のまたは後続のアーティファクトバージョンによって使用されていない場合に、関連するファイルをストレージから削除します。

このページのセクションでは、特定のアーティファクトバージョンを削除する方法、アーティファクトコレクションを削除する方法、エイリアスのあるアーティファクトとないアーティファクトを削除する方法などについて説明しています。TTLポリシーを使用して、W&Bからアーティファクトが削除されるタイミングをスケジュールできます。詳細については、[アーティファクトTTLポリシーによるデータ保持の管理]({{< relref path="./ttl.md" lang="ja" >}})を参照してください。

{{% alert %}}
TTLポリシーで削除がスケジュールされたアーティファクト、W&B SDKで削除されたアーティファクト、またはW&B App UIで削除されたアーティファクトは、まずソフトデリートされます。ソフトデリートされたアーティファクトは、ハードデリートされる前にガベージコレクションが行われます。
{{% /alert %}}

### アーティファクトバージョンを削除する

アーティファクトバージョンを削除するには：

1. アーティファクトの名前を選択します。これによりアーティファクトビューが拡張され、そのアーティファクトに関連付けられたすべてのアーティファクトバージョンが表示されます。
2. アーティファクトのリストから、削除したいアーティファクトバージョンを選択します。
3. ワークスペースの右側にあるケバブドロップダウンを選択します。
4. [削除]を選択します。

アーティファクトバージョンは、[delete()]({{< relref path="/ref/python/artifact#delete" lang="ja" >}})メソッドを使用してプログラムで削除することもできます。以下の例を参照してください。

### エイリアス付きの複数のアーティファクトバージョンを削除する

次のコード例は、エイリアスと関連づけられたアーティファクトを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、およびrun IDを指定します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

アーティファクトが1つ以上のエイリアスを持つ場合、`delete_aliases`パラメータをブール値の`True`に設定してエイリアスを削除します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # エイリアスを持つアーティファクトを削除するために
    # delete_aliases=Trueに設定
    artifact.delete(delete_aliases=True)
```

### 特定のエイリアスを持つ複数のアーティファクトバージョンを削除する

次のコードは、特定のエイリアスを持つ複数のアーティファクトバージョンを削除する方法を示しています。アーティファクトを作成したエンティティ、プロジェクト名、およびrun IDを指定します。削除ロジックを独自のものに置き換えてください。

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

以下のコードスニペットは、エイリアスを持たないアーティファクトのすべてのバージョンを削除する方法を示しています。`wandb.Api`で使用する`project`と`entity`キーにそれぞれプロジェクトとエンティティの名前を指定します。`<>`をアーティファクトの名前に置き換えてください。

```python
import wandb

# wandb.Apiメソッドを使用する際に、エンティティとプロジェクト名を指定します。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # タイプと名前を指定
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest'などのエイリアスを持たないバージョンをクリーンアップします。
    # 注：ここに任意の削除ロジックを入れることができます。
    if len(v.aliases) == 0:
        v.delete()
```

### アーティファクトコレクションを削除する

アーティファクトコレクションを削除するには：

1. 削除したいアーティファクトコレクションに移動して、その上にカーソルを置きます。
3. アーティファクトコレクション名の横にあるケバブドロップダウンを選択します。
4. [削除]を選択します。

アーティファクトコレクションは、[delete()]({{< relref path="/ref/python/artifact.md#delete" lang="ja" >}})メソッドを使ってプログラムで削除することもできます。`wandb.Api`で使用する`project`と`entity`キーにそれぞれプロジェクトとエンティティの名前を指定します。

```python
import wandb

# wandb.Apiメソッドを使用する際に、エンティティとプロジェクト名を指定します。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&Bのホスト方法に基づいてガベージコレクションを有効にする方法
ガベージコレクションは、W&Bの共有クラウドを使用している場合、デフォルトで有効になっています。W&Bをホストする方法に基づいて、ガベージコレクションを有効にするために追加の手順が必要になる場合があります。これには以下が含まれます：

* `GORILLA_ARTIFACT_GC_ENABLED`環境変数をtrueに設定します: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/object-versioning)や[Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning)のような他のストレージプロバイダーを使用している場合は、バケットバージョニングを有効にします。Azureを使用している場合は、[ソフトデリートを有効に](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview)します。

  {{% alert %}}
  Azureでのソフトデリートは、他のストレージプロバイダーにおけるバケットバージョニングに相当します。
  {{% /alert %}}

以下の表は、デプロイメントタイプに基づいてガベージコレクションを有効にするための要件を満たす方法を説明しています。

`X`は要件を満たす必要があることを示しています：

|                                                | 環境変数    | バージョニングの有効化 | 
| -----------------------------------------------| ------------------------| ----------------- | 
| 共有クラウド                                   |                         |                   | 
| [セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を使用した共有クラウド|                         | X                 | 
| 専用クラウド                                   |                         |                   | 
| [セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を使用した専用クラウド|                         | X                 | 
| カスタマー管理のクラウド                        | X                       | X                 | 
| カスタマー管理のオンプレミス                    | X                       | X                 |

{{% alert %}}注意
セキュアストレージコネクタは現在、Google Cloud PlatformとAmazon Web Servicesでのみ利用可能です。
{{% /alert %}}