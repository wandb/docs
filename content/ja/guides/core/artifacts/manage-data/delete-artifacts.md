---
title: アーティファクトを削除する
description: App UI で対話的にアーティファクトを削除するか、W&B SDK を使ってプログラムから削除できます。
menu:
  default:
    identifier: delete-artifacts
    parent: manage-data
---

App UI を使ってインタラクティブに、または W&B SDK を使ってプログラムから Artifacts を削除できます。Artifact を削除すると、W&B はその Artifact を *ソフトデリート* としてマークします。つまり、Artifact は削除予定としてマークされますが、ファイルはすぐにはストレージから削除されません。

Artifact の内容は、ソフトデリート（削除保留）状態のまま、定期的に実行されるガベージコレクションプロセスが削除対象となっている Artifacts をチェックするまで維持されます。ガベージコレクションプロセスは、Artifact やそれに紐付くファイルが過去や今後の他の Artifact バージョンで使われていなければ、関連するファイルをストレージから削除します。

このページの各セクションでは、特定の Artifact バージョンを削除する方法、Artifact コレクションを削除する方法、エイリアスの有無に関わらず Artifact を削除する方法などについて説明します。TTL ポリシーを使って、Artifacts が W&B から削除されるタイミングをスケジューリングできます。詳細は [Artifact の TTL ポリシーでデータ保持を管理する]({{< relref "./ttl.md" >}}) を参照してください。

{{% alert %}}
TTL ポリシーで削除がスケジューリングされた Artifact、W&B SDK で削除された Artifact、または W&B App UI で削除された Artifact は、まずソフトデリートされます。ソフトデリートされた Artifact は、ハードデリートされる前にガベージコレクションを受けます。
{{% /alert %}}

### Artifact バージョンを削除する

Artifact バージョンを削除するには:

1. Artifact の名前を選択します。これで Artifact ビューが展開され、その Artifact に関連付いた全バージョンが表示されます。
2. Artifact 一覧から、削除したい Artifact バージョンを選択します。
3. Workspace の右側にあるケバブドロップダウンをクリックします。
4. 削除を選択します。

Artifact バージョンは [delete()]({{< relref "/ref/python/sdk/classes/artifact.md#delete" >}}) メソッドを使ってプログラムから削除することもできます。以下の例を参照してください。

### エイリアス付き Artifact バージョンを複数削除する

次のコード例は、エイリアスが付与されている Artifact を削除する方法を示しています。Entity、プロジェクト名、Artifact を作成した run ID を指定します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

Artifact に 1 つ以上のエイリアスがある場合、それも一緒に削除したい場合は `delete_aliases` パラメータを True に設定します。

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # エイリアス付き Artifact も削除したい場合は
    # delete_aliases=True を指定します
    artifact.delete(delete_aliases=True)
```

### 特定エイリアス付き Artifact バージョンを複数削除する

次のコードは、特定のエイリアスを持つ複数の Artifact バージョンを削除する方法を示しています。Entity、プロジェクト名、Artifact を作成した run ID を指定してください。また、削除ロジックは用途に合わせて書き換えてください。

```python
import wandb

runs = api.run("entity/project_name/run_id")

# エイリアス 'v3' と 'v4' を持つ Artifact を削除
for artifact_version in runs.logged_artifacts():
    # ご自身の削除条件に置き換えてください。
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### エイリアスを持っていない Artifact バージョンをすべて削除する

次のコードスニペットは、エイリアスを持たない全ての Artifact バージョンを削除する方法です。`project` および `entity` キーには、それぞれプロジェクト名と Entity 名を指定してください。`<>` の部分は、ご自身の Artifact 名などに置き換えてください。

```python
import wandb

# wandb.Api メソッドを使う際は entity および project 名を指定してください。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # 種類と名前を指定
for v in api.artifact_versions(artifact_type, artifact_name):
    # 'latest' などエイリアスがないバージョンをクリーンアップ
    # 注: この部分は用途に合わせて削除ロジックを記述できます。
    if len(v.aliases) == 0:
        v.delete()
```

### Artifact コレクションを削除する

Artifact コレクションを削除するには:

1. 削除したい Artifact コレクションに移動し、カーソルを合わせます。
3. コレクション名の横のケバブドロップダウンをクリックします。
4. 削除を選択します。

Artifact コレクションも [delete()]({{< relref "/ref/python/sdk/classes/artifact.md#delete" >}}) メソッドでプログラムから削除できます。`wandb.Api` の `project` と `entity` キーにプロジェクト名と Entity 名を指定してください。

```python
import wandb

# wandb.Api メソッドを使う際は entity および project 名を指定してください。
api = wandb.Api(overrides={"project": "project", "entity": "entity"})
collection = api.artifact_collection(
    "<artifact_type>", "entity/project/artifact_collection_name"
)
collection.delete()
```

## W&B の提供形態ごとのガベージコレクション有効化方法

W&B の共有クラウドを利用している場合、ガベージコレクションはデフォルトで有効です。ご利用の W&B ホスティング形態に応じて、追加設定が必要になる場合があります。例えば以下です：

* `GORILLA_ARTIFACT_GC_ENABLED` 環境変数を true に設定: `GORILLA_ARTIFACT_GC_ENABLED=true`
* [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/object-versioning) などや、[Minio](https://min.io/docs/minio/linux/administration/object-management/object-versioning.html#enable-bucket-versioning) のような他のストレージプロバイダーを使う場合は、バケットのバージョン管理を有効にしてください。Azure の場合は、[ソフトデリートを有効化](https://learn.microsoft.com/azure/storage/blobs/soft-delete-blob-overview) してください。
  {{% alert %}}
  Azure でのソフトデリートは、他のストレージプロバイダーのバケットバージョン管理と同等です。
  {{% /alert %}}

以下の表は、ご利用のデプロイメントタイプに応じてガベージコレクション有効化に必要な条件を示します。

`X` はその要件を満たす必要があることを示します。

|                                                | 環境変数設定           | バージョン管理有効化 |
| -----------------------------------------------| ---------------------- | ------------------- |
| 共有クラウド                                   |                        |                     |
| 共有クラウド + [Secure Storage Connector]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) |                        | X                   |
| 専用クラウド                                   |                        |                     |
| 専用クラウド + [Secure Storage Connector]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) |                        | X                   |
| カスタマー管理クラウド                         | X                      | X                   |
| カスタマー管理オンプレミス                     | X                      | X                   |

{{% alert %}}note
Secure Storage Connector は現在、Google Cloud Platform および Amazon Web Services のみ対応しています。
{{% /alert %}}