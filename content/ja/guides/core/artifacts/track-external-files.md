---
title: 外部ファイルをトラッキングする
description: 外部バケット、HTTPファイルサーバー、または NFS 共有に保存されたファイルをトラッキングします。
menu:
  default:
    identifier: track-external-files
    parent: artifacts
weight: 7
---

**リファレンスアーティファクト** を使うことで、CoreWeave AI Object Storage や Amazon Simple Storage Service (Amazon S3) バケット、GCS バケット、Azure Blob、HTTP ファイルサーバー、NFS 共有など、W&B サーバー外に保存されたファイルをトラッキング・利用できます。

W&B はオブジェクトの ETag やサイズなど、オブジェクトに関するメタデータをログします。バケットでオブジェクトのバージョン管理が有効になっている場合は、バージョン ID も記録されます。

{{% alert %}}
外部ファイルをトラッキングしないアーティファクトをログすると、W&B はアーティファクトのファイルを W&B サーバーに保存します。これは W&B Python SDK でアーティファクトをログする際のデフォルト動作です。

ファイルやディレクトリーを W&B サーバーに保存する方法については [Artifacts クイックスタート]({{< relref "/guides/core/artifacts/artifacts-walkthrough" >}}) をご覧ください。
{{% /alert %}}

以下に、リファレンスアーティファクトの作成方法を説明します。

## 外部バケット内のアーティファクトをトラッキングする

W&B Python SDK を使用して、W&B 外部に保存されたファイルへのリファレンスをトラッキングできます。

1. `wandb.init()` で run を初期化する。
2. `wandb.Artifact()` でアーティファクトオブジェクトを作成する。
3. アーティファクトオブジェクトの `add_reference()` メソッドを使ってバケットパスへの参照を指定する。
4. `run.log_artifact()` でアーティファクトのメタデータをログする。

```python
import wandb

# W&B の run を初期化
run = wandb.init()

# アーティファクトオブジェクトを作成
artifact = wandb.Artifact(name="name", type="type")

# バケットパスへのリファレンスを追加
artifact.add_reference(uri = "uri/to/your/bucket/path")

# アーティファクトのメタデータをログ
run.log_artifact(artifact)
run.finish()
```

例えば、あなたのバケットのディレクトリ構造が次のようになっているとします：

```text
s3://my-bucket

|datasets/
  |---- mnist/
|models/
  |---- cnn/
```

`datasets/mnist/` ディレクトリには画像データセットが含まれています。`wandb.Artifact.add_reference()` を使ってこのディレクトリ全体をデータセットとしてトラッキングできます。以下のコードは、アーティファクトオブジェクトの `add_reference()` メソッドを使い、`mnist:latest` というリファレンスアーティファクトを作成します。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact(name="mnist", type="dataset")
artifact.add_reference(uri="s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
run.finish()
```

W&B アプリ内では、リファレンスアーティファクトの内容をファイルブラウザで確認したり、[依存グラフ全体を探索]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph" >}}) したり、アーティファクトのバージョン履歴を調べたりできます。W&B アプリは、実際のデータがアーティファクト内に含まれていないため、画像や音声などのリッチメディアは表示できません。

{{% alert %}}
W&B Artifacts は MinIO など、Amazon S3 互換インターフェースをサポートしています。以下のスクリプトは `AWS_S3_ENDPOINT_URL` 環境変数で MinIO サーバーを指定すれば、そのまま MinIO でも動作します。
{{% /alert %}}

{{% alert color="secondary" %}}
デフォルトでは、オブジェクトのプレフィックスを追加する際、W&B は最大 10,000 オブジェクトの制限があります。`add_reference()` を呼ぶ際に `max_objects=` を指定することでこの上限を調整できます。
{{% /alert %}}

## 外部バケットからアーティファクトをダウンロードする

W&B は、リファレンスアーティファクトをダウンロードする際、アーティファクト記録時のメタデータに基づいてバケットからファイルを取得します。バケットでオブジェクトのバージョン管理を有効にしている場合、アーティファクトがログされた時点のバージョンが取得されます。バケットの内容が更新されても、各モデルがどのデータでトレーニングされたか正確に指し示すことができます。アーティファクトはトレーニング run 時点のバケットのスナップショットを保存しているためです。

以下のコード例は、リファレンスアーティファクトのダウンロード方法を示しています。リファレンスかどうかに関わらず、アーティファクトのダウンロード API は同じです：

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

{{% alert %}}
ワークフローの中でファイルを書き換える場合は、ストレージバケットで「オブジェクトバージョン管理」を有効にすることを推奨します。バージョン管理を有効にしておけば、ファイルが上書きされても古いバージョンも保持されるため、リファレンス付きアーティファクトの整合性が保たれます。

ユースケースに応じて、バージョン管理の有効化方法を参照してください：[AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/using-object-versioning#set)、[Azure](https://learn.microsoft.com/azure/storage/blobs/versioning-enable)。
{{% /alert %}}

### 外部リファレンスの追加とダウンロード例

次のサンプルコードは、データセットを Amazon S3 バケットにアップロードし、リファレンスアーティファクトでトラッキングし、その後ダウンロードする例です：

```python
import boto3
import wandb

run = wandb.init()

# トレーニング処理...

s3_client = boto3.client("s3")
s3_client.upload_file(file_name="my_model.h5", bucket="my-bucket", object_name="models/cnn/my_model.h5")

# モデルアーティファクトをログ
model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

後でこのモデルアーティファクトをダウンロードできます。アーティファクト名とタイプを指定してください：

```python
import wandb

run = wandb.init()
artifact = run.use_artifact(artifact_or_name = "cnn", type="model")
datadir = artifact.download()
```

{{% alert %}}
GCP や Azure でリファレンスによるアーティファクトトラッキングをエンドツーエンドで解説したレポートをご覧ください：

* [Guide to Tracking Artifacts by Reference with GCP](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Working with Reference Artifacts in Microsoft Azure](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}

## クラウドストレージの認証情報

W&B は、ご利用中のクラウドプロバイダに応じて認証情報を探すデフォルトの仕組みを利用します。各クラウドプロバイダの認証情報については、下記ドキュメントをご参照ください：

| クラウドプロバイダ | 認証情報ドキュメント |
| -------------- | ------------------------- |
| CoreWeave AI Object Storage | [CoreWeave AI Object Storage documentation](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/manage-access-keys/cloud-console-tokens) |
| AWS            | [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure documentation](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWS を利用していて、バケットが設定されたユーザーのデフォルトリージョン以外に存在する場合は、`AWS_REGION` 環境変数をバケットのリージョンに合わせて設定してください。

{{% alert color="secondary" %}}
バケットの CORS 設定によっては、画像・音声・動画・点群などのリッチメディアが App UI で正常に表示できない場合があります。バケットの CORS 設定で **app.wandb.ai** を許可リストに追加することで、App UI でこれらのリッチメディアが正しく表示されるようになります。

もしリッチメディアが App UI で表示できない場合は、バケットの CORS ポリシーで `app.wandb.ai` が許可リストに登録されていることを確認してください。
{{% /alert %}}

## ファイルシステム内のアーティファクトをトラッキングする

データセットへの高速なアクセスを実現する一般的なパターンとして、すべてのトレーニングジョブ実行マシンから NFS のマウントポイントを利用する方法があります。これはクラウドストレージバケットよりもシンプルなことが多く、トレーニングスクリプトからはローカルファイルと同様に扱えます。Artifacts でもこの使いやすさを継承しており、マウントされた／されていないファイルシステムも同様にリファレンストラッキングできます。

例えば `/mount` にマウントされたファイルシステムの構造が以下のような場合：

```bash
mount
|datasets/
		|-- mnist/
|models/
		|-- cnn/
```

`mnist/` の中には画像コレクション (データセット) があります。これをアーティファクトとしてトラッキングできます：

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```
{{% alert color="secondary" %}}
デフォルトでは、ディレクトリーへのリファレンス追加時に W&B は最大 10,000 ファイルの制限があります。`add_reference()` を呼ぶ際 `max_objects=` を指定してこの上限を調整できます。
{{% /alert %}}

URL 中のスラッシュ 3 つに注意してください。最初の部分はファイルシステムリファレンスを示す `file://` で、次にデータセットへのパス(`/mount/datasets/mnist/`)が続きます。

作成されるアーティファクト `mnist:latest` は通常のアーティファクトと同じように見え、振る舞います。唯一の違いは、ファイルのサイズや MD5 チェックサムといったメタデータのみで構成されており、実際のファイル自体はあなたのシステムから移動しないことです。

このアーティファクトは、通常のアーティファクトと同様に操作できます。UI 上では、リファレンスアーティファクトの内容閲覧や依存グラフの確認、バージョン履歴の参照も可能です。ただし、データ自体がアーティファクトに含まれない都合上、UI では画像・音声などのリッチメディアは表示できません。

リファレンスアーティファクトのダウンロード：

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

ファイルシステムリファレンスの場合、`download()` 操作は参照されているパスからファイルをコピーし、アーティファクトディレクトリーを生成します。上記例では `/mount/datasets/mnist` の内容が `artifacts/mnist:v0/` ディレクトリーにコピーされます。もしアーティファクトが上書きされたファイルへの参照を含む場合、`download()` はエラーになります。これは、アーティファクトが再構成できなくなったためです。

すべてをまとめると、次のようなコードでマウント済みファイルシステム上のデータセットをトレーニングジョブに取り込んでトラッキングできます：

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# この run のインプットとしてアーティファクトをトラッキング
# ディレクトリー下のファイルに変更があった場合のみ新バージョンがログされます
run.use_artifact(artifact)

artifact_dir = artifact.download()

# ここでトレーニング処理...
```

モデルをトラッキングする場合は、トレーニングスクリプトがモデルファイルをマウントポイントに書き出した後、モデルアーティファクトをログします：

```python
import wandb

run = wandb.init()

# トレーニング処理...

# モデルをディスクに書き出し

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```