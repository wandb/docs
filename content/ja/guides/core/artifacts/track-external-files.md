---
title: 外部ファイルを記録する
description: 外部のバケット、HTTP ファイルサーバー、または NFS 共有に保存されたファイルを追跡します。
menu:
  default:
    identifier: ja-guides-core-artifacts-track-external-files
    parent: artifacts
weight: 7
---

**参照 Artifacts** を使用して、W&B サーバーの外部に保存されているファイルを追跡および使用します。例えば、CoreWeave AI Object Storage、Amazon Simple Storage Service (Amazon S3) バケット、GCS バケット、Azure Blob、HTTP ファイルサーバー、または NFS 共有などです。
W&B は、オブジェクトの ETag やサイズなど、そのオブジェクトに関するメタデータをログに記録します。バケットでオブジェクトのバージョン管理が有効になっている場合、バージョン ID もログに記録されます。
{{% alert %}}
外部ファイルを追跡しない Artifacts をログに記録する場合、W&B は Artifact のファイルを W&B サーバーに保存します。これは、W&B Python SDK で Artifacts をログに記録する際のデフォルトの振る舞いです。
代わりに W&B サーバーにファイルとディレクトリーを保存する方法については、[Artifacts クイックスタート]({{< relref path="/guides/core/artifacts/artifacts-walkthrough" lang="ja" >}}) を参照してください。
{{% /alert %}}
以下では、参照 Artifacts を構築する方法について説明します.
## 外部バケットの Artifact を追跡する
W&B Python SDK を使用して、W&B の外部に保存されているファイルへの参照を追跡します。
1. `wandb.init()` を使用して run を初期化します。
2. `wandb.Artifact()` を使用して Artifact オブジェクトを作成します。
3. Artifact オブジェクトの `add_reference()` メソッドを使用して、バケットパスへの参照を指定します。
4. `run.log_artifact()` を使用して Artifact のメタデータをログに記録します。
```python
import wandb

# W&B run を初期化します
run = wandb.init()

# Artifact オブジェクトを作成します
artifact = wandb.Artifact(name="name", type="type")

# バケットパスへの参照を追加します
artifact.add_reference(uri = "uri/to/your/bucket/path")

# Artifact のメタデータをログに記録します
run.log_artifact(artifact)
run.finish()
```
バケットが以下のディレクトリー構造を持っているとします。
```text
s3://my-bucket

|datasets/
  |---- mnist/
|models/
  |---- cnn/
```
`datasets/mnist/` ディレクトリーには画像のコレクションが含まれています。`wandb.Artifact.add_reference()` を使用して、このディレクトリーをデータセットとして追跡します。以下のコードサンプルは、Artifact オブジェクトの `add_reference()` メソッドを使用して、参照 Artifact `mnist:latest` を作成します。
```python
import wandb

run = wandb.init()
artifact = wandb.Artifact(name="mnist", type="dataset")
artifact.add_reference(uri="s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
run.finish()
```
W&B App 内で、ファイルブラウザーを使用して参照 Artifact のコンテンツを閲覧し、[完全な依存関係グラフを探索]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}) し、Artifact のバージョン履歴をスキャンできます。データ自体は Artifact に含まれていないため、W&B App は画像や音声などのリッチメディアをレンダリングしません。
{{% alert %}}
W&B Artifacts は、CoreWeave Storage や MinIO を含む、あらゆる Amazon S3 互換インターフェースをサポートしています。`AWS_S3_ENDPOINT_URL` 環境変数を CoreWeave Storage または MinIO サーバーを指すように設定すると、以下で説明するスクリプトは両方のプロバイダーでそのまま動作します。
{{% /alert %}}
{{% alert color="secondary" %}}
デフォルトでは、W&B はオブジェクトプレフィックスを追加する際に 10,000 オブジェクトの制限を課します。`add_reference()` を呼び出すときに `max_objects=` を指定することで、この制限を調整できます。
{{% /alert %}}
## 外部バケットから Artifact をダウンロードする
W&B は、Artifact がログに記録されたときに記録されたメタデータを使用して参照 Artifact をダウンロードする際に、基盤となるバケットからファイルを取得します。バケットでオブジェクトのバージョン管理が有効になっている場合、W&B は Artifact がログに記録された時点のファイルの状態に対応するオブジェクトバージョンを取得します。バケットのコンテンツを更新しても、特定のモデルがトレーニングされたデータの正確なバージョンを常に指すことができます。これは、Artifact がトレーニング run 中のバケットのスナップショットとして機能するためです。
以下のコードサンプルは、参照 Artifact をダウンロードする方法を示しています。Artifacts をダウンロードするための API は、参照 Artifacts と非参照 Artifacts の両方で同じです。
```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```
{{% alert %}}
ワークフローの一部としてファイルを上書きする場合、W&B はストレージバケットで「オブジェクトのバージョン管理」を有効にすることを推奨します。バケットでバージョン管理が有効になっている場合、上書きされたファイルへの参照を持つ Artifacts は、古いオブジェクトバージョンが保持されているため、引き続き無傷です。
ユースケースに基づいて、オブジェクトのバージョン管理を有効にする手順を読んでください: [AWS](https://docs.aws.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/using-object-versioning#set)、[Azure](https://learn.microsoft.com/azure/storage/blobs/versioning-enable)。
{{% /alert %}}
### 外部参照の追加とダウンロードの例
以下のコードサンプルは、データセットを Amazon S3 バケットにアップロードし、参照 Artifact で追跡し、その後ダウンロードする方法を示しています。
```python
import boto3
import wandb

run = wandb.init()

# ここでトレーニング...

s3_client = boto3.client("s3")
s3_client.upload_file(file_name="my_model.h5", bucket="my-bucket", object_name="models/cnn/my_model.h5")

# モデル Artifact をログに記録します
model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```
後で、モデル Artifact をダウンロードできます。Artifact の名前とそのタイプを指定します。
```python
import wandb

run = wandb.init()
artifact = run.use_artifact(artifact_or_name = "cnn", type="model")
datadir = artifact.download()
```
{{% alert %}}
GCP または Azure で参照によって Artifacts を追跡する方法に関するエンドツーエンドのウォークスルーについては、以下のレポートを参照してください。
*   [GCP を使用して参照によって Artifacts を追跡するガイド](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
*   [Microsoft Azure で参照 Artifacts を使用する](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}
## クラウドストレージの認証情報
W&B は、使用するクラウドプロバイダーに基づいて認証情報を検索するデフォルトのメカニズムを使用します。使用される認証情報について詳しくは、クラウドプロバイダーのドキュメントを参照してください。
| クラウドプロバイダー | 認証情報ドキュメント |
| -------------- | ------------------------- |
| CoreWeave AI Object Storage | [CoreWeave AI Object Storage documentation](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/manage-access-keys/cloud-console-tokens) |
| AWS            | [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure documentation](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |
AWS の場合、バケットが設定されたユーザーのデフォルトリージョンにない場合、`AWS_REGION` 環境変数をバケットリージョンと一致するように設定する必要があります。
{{% alert color="secondary" %}}
画像、音声、ビデオ、点群などのリッチメディアは、バケットの CORS 設定によっては App UI でレンダリングされない場合があります。バケットの CORS 設定で **app.wandb.ai** を許可リストに追加すると、App UI でそのようなリッチメディアが適切にレンダリングされます。
画像、音声、ビデオ、点群などのリッチメディアが App UI でレンダリングされない場合は、`app.wandb.ai` がバケットの CORS ポリシーで許可リストに追加されていることを確認してください。
{{% /alert %}}
## ファイルシステムの Artifact を追跡する
データセットへの高速アクセスにおけるもう一つの一般的なパターンは、トレーニングジョブを実行しているすべてのマシンでリモートファイルシステムへの NFS マウントポイントを公開することです。これはクラウドストレージバケットよりもさらにシンプルな解決策になりえます。なぜなら、トレーニングスクリプトの観点からは、ファイルがローカルファイルシステムに存在するように見えるからです。幸いなことに、その使いやすさは、マウントされているかどうかにかかわらず、Artifacts を使用してファイルシステムへの参照を追跡することにも及びます。
`/mount` に以下の構造でマウントされたファイルシステムがあるとします。
```bash
mount
|datasets/
		|-- mnist/
|models/
		|-- cnn/
```
`mnist/` 内には、画像のコレクションであるデータセットがあります。Artifact を使用して追跡できます。
```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```
{{% alert color="secondary" %}}
デフォルトでは、W&B はディレクトリーへの参照を追加する際に 10,000 ファイルの制限を課します。`add_reference()` を呼び出すときに `max_objects=` を指定することで、この制限を調整できます。
{{% /alert %}}
URL のトリプルスラッシュに注意してください。最初のコンポーネントは、ファイルシステム参照の使用を示す `file://` プレフィックスです。2 番目のコンポーネントは、データセットへのパス `/mount/datasets/mnist/` を開始します。
結果として生成される Artifact `mnist:latest` は、通常の Artifact と同じように見え、動作します。唯一の違いは、Artifact がファイルのサイズや MD5 チェックサムなどのメタデータのみで構成されていることです。ファイル自体がシステムから離れることはありません。
この Artifact は、通常の Artifact と同じように操作できます。UI では、ファイルブラウザーを使用して参照 Artifact のコンテンツを閲覧し、完全な依存関係グラフを探索し、Artifact のバージョン履歴をスキャンできます。ただし、データ自体が Artifact に含まれていないため、UI は画像や音声などのリッチメディアをレンダリングできません。
参照 Artifact をダウンロードする:
```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```
ファイルシステム参照の場合、`download()` 操作は参照されたパスからファイルをコピーして Artifact ディレクトリーを構築します。上記の例では、`/mount/datasets/mnist` のコンテンツがディレクトリー `artifacts/mnist:v0/` にコピーされます。Artifact が上書きされたファイルへの参照を含んでいる場合、Artifact を再構築できなくなるため、`download()` はエラーをスローします。
これらすべてをまとめると、マウントされたファイルシステム上のデータセットを追跡し、それをトレーニングジョブに供給するために以下のコードを使用できます。
```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# Artifact を追跡し、それをこの run の入力としてマークします。
# 新しい Artifact バージョンは
# ディレクトリー内のファイルが変更された場合にのみ
# ログに記録されます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# ここでトレーニングを実行します...
```
モデルを追跡するには、トレーニングスクリプトがモデルファイルをマウントポイントに書き込んだ後、モデル Artifact をログに記録します。
```python
import wandb

run = wandb.init()

# ここでトレーニング...

# モデルをディスクに書き込みます

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```