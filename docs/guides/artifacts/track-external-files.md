---
description: W&Bの外部に保存されたファイルを追跡します。例えば、Amazon S3 バケット、GCS バケット、HTTPファイルサーバー、あるいはNFS共有などです。
displayed_sidebar: default
---


# Track external files

<head>
	<title>参照Artifactsを使用して外部ファイルをトラッキングする</title>
</head>

**Reference Artifacts** を使用して、Amazon S3バケット、GCSバケット、Azure Blob、HTTPファイルサーバー、またはNFSシェアのようなW&Bシステム外に保存されたファイルをトラッキングします。W&B [CLI](https://docs.wandb.ai/ref/cli) を使用して、[W&B Run](https://docs.wandb.ai/ref/python/run)の外でArtifactsをログします。

### Runの外でArtifactsをログする

W&BはRunの外でArtifactsをログするときにRunを作成します。各ArtifactはRunに属し、そのRunはProjectに属します。また、Artifactのバージョンはコレクションに属し、タイプを持ちます。

[`wandb artifact put`](https://docs.wandb.ai/ref/cli/wandb-artifact/wandb-artifact-put) コマンドを使って、W&B runの外でArtifactをW&Bサーバーにアップロードします。Artifactを属させたいProjectの名前とArtifactの名前（`project/artifact_name`）を指定します。オプションでタイプ（`TYPE`）を指定します。以下のコードスニペットの`PATH`をアップロードしたいArtifactのファイルパスに置き換えてください。

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

指定したProjectが存在しない場合、W&Bは新しいProjectを作成します。Artifactのダウンロード方法については、[Download and use artifacts](https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact) を参照してください。

## W&Bの外でArtifactsをトラッキングする

W&B Artifactsを使用してデータセットのバージョン管理やモデルリネージを行い、**Reference Artifacts** を使用してW&Bサーバーの外に保存されたファイルをトラッキングします。このモードでは、Artifactはファイルのメタデータ（URL、サイズ、チェックサムなど）のみを保存します。基となるデータはシステムから出ることはありません。ファイルやディレクトリーをW&Bサーバーに保存する方法については、[Quick start](https://docs.wandb.ai/guides/artifacts/artifacts-walkthrough) を参照してください。

以下に、参照Artifactsを構築する方法とそれらをワークフローに組み込む最適な方法を説明します。

### Amazon S3 / GCS / Azure Blob Storage 参照

W&B Artifactsを使用してクラウドストレージバケット内のデータセットやモデルのバージョン管理を行います。Artifactの参照を使えば、既存のストレージレイアウトを変更せずにバケットの上にトラッキングをシームレスにレイヤー化できます。

Artifactsは基となるクラウドストレージベンダー（AWS、GCP、Azureなど）を抽象化します。次のセクションで説明する情報は、Amazon S3、Google Cloud Storage、Azure Blob Storageに一様に適用されます。

:::info
W&B Artifactsは、MinIOを含むすべてのAmazon S3互換インターフェースをサポートします！以下のスクリプトは、AWS_S3_ENDPOINT_URL環境変数をMinIOサーバーに設定するだけで動作します。
:::

次のような構造を持つバケットがあるとします：

```bash
s3://my-bucket
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/`の下にはイメージのコレクションであるデータセットがあります。これをArtifactでトラッキングしましょう：

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```
:::caution
デフォルトでは、オブジェクトのプレフィックスを追加する際にW&Bは10,000オブジェクトの制限を課しています。この制限は`add_reference`への呼び出しで`max_objects=`を指定して調整できます。
:::

新しい参照Artifact `mnist:latest`は通常のArtifactと同様に見え、動作します。唯一の違いは、ArtifactがS3/GCS/Azureオブジェクトに関するETag、サイズ、およびバージョンID（バケットでオブジェクトバージョン管理が有効になっている場合）といったメタデータのみで構成されていることです。

W&Bは使用するクラウドプロバイダーに基づいて資格情報を探すデフォルトのメカニズムを使用します。クラウドプロバイダーの資格情報について詳細は、それぞれのドキュメントを参照してください：

| クラウドプロバイダー | 資格情報ドキュメント |
| --------------------- | ------------------------- |
| AWS            | [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure documentation](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWSの場合、バケットが配置されている地域が構成済みのユーザーのデフォルトの地域と異なる場合、`AWS_REGION`環境変数をバケットの地域に設定する必要があります。

このArtifactを通常のArtifactと同様に操作します。App UIでは、ファイルブラウザを使用して参照Artifactの内容を閲覧し、完全な依存関係グラフを調査し、Artifactのバージョン履歴をスキャンできます。

:::caution
バケットのCORS設定に応じて、画像、音声、ビデオ、ポイントクラウドなどのリッチメディアがApp UIで適切にレンダリングされない場合があります。バケットのCORS設定で **app.wandb.ai** を許可リストに追加することで、App UIでリッチメディアを正しくレンダリングできます。

プライベートバケットの場合、パネルがApp UIでレンダリングに失敗することがあります。会社にVPNがある場合、バケットのアクセスポリシーを更新してVPN内のIPを許可リストに追加できます。
:::

### 参照Artifactをダウンロードする

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

参照Artifactをダウンロードすると、Artifactがログされたときに記録されたメタデータを使用して、W&Bは基となるバケットからファイルを取得します。バケットにオブジェクトバージョン管理が有効になっている場合、W&BはArtifactがログされた時点のファイル状態に対応するオブジェクトバージョンを取得します。これにより、バケットの内容が変化しても、トレーニング時にモデルが使用したデータの正確なバージョンを指し示すことができます。

:::info
ワークフローの一環としてファイルを上書きする場合、W&Bはストレージバケットに`オブジェクトバージョン管理`を有効にすることを推奨します。バージョン管理が有効なバケットでは、参照先ファイルが上書きされても、古いオブジェクトバージョンは保持されるため、Artifactsは依然として無傷です。

ユースケースに基づいて、オブジェクトバージョン管理の有効化手順を確認してください：[AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/using-object-versioning#set), [Azure](https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-enable)。
:::

### まとめ

以下のコード例は、Amazon S3、GCS、またはAzureでのデータセットをトラッキングし、トレーニングジョブにフィードするためのシンプルなワークフローを示しています：

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")

# Artifactをトラッキングし、このRunの入力として
# マークします。バケット内のファイルが変わった場合にのみ
# 新しいArtifactバージョンがログされます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# トレーニングをここで実行...
```

モデルをトラッキングするには、トレーニングスクリプトがモデルファイルをバケットにアップロードした後でモデルArtifactをログします：

```python
import boto3
import wandb

run = wandb.init()

# トレーニング...

s3_client = boto3.client("s3")
s3_client.upload_file("my_model.h5", "my-bucket", "models/cnn/my_model.h5")

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

:::info
GCPまたはAzureのArtifactsを参照してトラッキングする方法のエンドツーエンドの説明については、次のレポートを参照してください：

* [Guide to Tracking Artifacts by Reference](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Working with Reference Artifacts in Microsoft Azure](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
:::

### ファイルシステム参照

データセットへの迅速なアクセスのためのもう一つの一般的なパターンは、NFSマウントポイントをすべてのトレーニングジョブを実行するマシンに公開することです。これはクラウドストレージバケットよりもシンプルなソリューションとなる可能性があります。トレーニングスクリプトの観点からは、ファイルはローカルのファイルシステムに置かれているように見えるからです。この使いやすさは、マウントされたファイルシステムへの参照をトラッキングするためのArtifactsを使用する場合でも同様です。

次のような構造を持つマウントされたファイルシステムがあるとします：

```bash
mount
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/`の下にはイメージのコレクションであるデータセットがあります。これをArtifactでトラッキングしましょう：

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```

デフォルトでは、W&Bはディレクトリーへの参照を追加する際に10,000ファイルの制限を課しています。この制限は`add_reference`への呼び出しで`max_objects=`を指定して調整できます。

URL中の三重スラッシュに注意してください。最初の部分はファイルシステム参照の使用を示す `file://` プレフィックスです。次にデータセットへのパスである `/mount/datasets/mnist/` が続きます。

結果として得られるArtifact `mnist:latest` は通常のArtifactと同様に見え、動作します。唯一の違いは、ArtifactがファイルのサイズやMD5チェックサムなどのメタデータのみで構成されていることです。ファイル自体はシステムから離れることはありません。

このArtifactを通常のArtifactと同様に操作できます。UIでは、ファイルブラウザを使用して参照Artifactの内容を閲覧し、完全な依存関係グラフを調査し、Artifactのバージョン履歴をスキャンできます。ただし、データがArtifact自体に含まれていないため、画像、音声などのリッチメディアはUIでレンダリングできません。

参照Artifactのダウンロードは簡単です：

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

ファイルシステム参照の場合、`download()`操作は参照パスからファイルをコピーしてArtifactディレクトリーを構築します。上記の例では、`/mount/datasets/mnist`の内容が`artifacts/mnist:v0/`ディレクトリーにコピーされます。Artifactが上書きされたファイルへの参照を含む場合、`download()`はエラーをスローします。Artifactを再構築できなくなったためです。

すべてをまとめて、マウントされたファイルシステムの下でデータセットをトラッキングし、トレーニングジョブにフィードするためのシンプルなワークフローは次のとおりです：

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# Artifactをトラッキングし、このRunの入力として
# マークします。ディレクトリー下のファイルが変更された場合にのみ
# 新しいArtifactバージョンがログされます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# トレーニングをここで実行...
```

モデルをトラッキングするには、トレーニングスクリプトがモデルファイルをマウントポイントに書き込んだ後でモデルArtifactをログします：

```python
import wandb

run = wandb.init()

# トレーニング...

# モデルをディスクに書き込み

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```
