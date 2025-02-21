---
title: Track external files
description: W&B の外で保存されたファイルを追跡します。例えば、Amazon S3 バケット、 GCS バケット、HTTP ファイル サーバー、または
  NFS 共有などです。
menu:
  default:
    identifier: ja-guides-core-artifacts-track-external-files
    parent: artifacts
weight: 7
---

**reference artifacts** を利用して、W&B システム外に保存されたファイルをトラッキングしましょう。例えば、Amazon S3 バケット、GCS バケット、Azure blob、HTTP ファイルサーバー、または NFS 共有などが対象です。W&B [CLI]({{< relref path="/ref/cli" lang="ja" >}})を使用して、[W&B Run]({{< relref path="/ref/python/run" lang="ja" >}})の外部でアーティファクトをログします。

### Run の外でアーティファクトをログする

W&B は、run の外でアーティファクトをログする際に run を作成します。各アーティファクトは、run に属し、その run はプロジェクトに属します。アーティファクト (バージョン) はコレクションにも属し、タイプを持ちます。

[`wandb artifact put`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put" lang="ja" >}})コマンドを使用して、W&B run の外でアーティファクトを W&B サーバーにアップロードします。アーティファクトを属させるプロジェクト名とアーティファクト名 (`project/artifact_name`) を指定します。タイプ (`TYPE`) をオプションで指定します。以下のコードスニペット内の `PATH` を、アップロードしたいアーティファクトのファイルパスに置き換えます。

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

指定したプロジェクトが存在しない場合、W&B は新しいプロジェクトを作成します。アーティファクトのダウンロード方法については、[Download and use artifacts]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}})を参照してください。

## W&B の外でアーティファクトをトラッキングする

W&B Artifacts を使用してデータセットのバージョン管理やモデルリネージを行い、**reference artifacts** を使用して W&B サーバー外に保存されたファイルをトラッキングします。このモードでは、アーティファクトはファイルのメタデータ (URL、サイズ、チェックサムなど) のみを保存します。基礎データはシステムから離れることはありません。ファイルやディレクトリを W&B サーバーに保存する方法については、[Quick start]({{< relref path="/guides/core/artifacts/artifacts-walkthrough" lang="ja" >}}) を参照してください。

以下は、reference artifacts を構築する方法と、それらをワークフローに最適に組み込む方法について説明しています。

### Amazon S3 / GCS / Azure Blob Storage の参照

W&B Artifacts を使用して、クラウドストレージバケット内の参照をトラッキングし、データセットやモデルのバージョン管理を行います。アーティファクト参照を使用して、既存のストレージレイアウトに変更を加えることなく、バケット上で追跡をレイヤー化します。

Artifacts は、基礎となるクラウドストレージのベンダー (AWS、GCP、Azure など) を抽象化します。次節で説明する情報は、Amazon S3、Google Cloud Storage、Azure Blob Storage に均一に適用されます。

{{% alert %}}
W&B Artifacts は、MinIO を含む Amazon S3 互換のすべてのインターフェイスをサポートしています。以下のスクリプトは、`AWS_S3_ENDPOINT_URL` 環境変数を MinIO サーバーに向けて設定するだけで、そのまま機能します。
{{% /alert %}}

以下の構造を持つバケットがあると仮定します:

```bash
s3://my-bucket
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` の下にはデータセットである画像のコレクションがあります。これをアーティファクトでトラッキングしましょう:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```
{{% alert color="secondary" %}}
デフォルトでは、W&B はオブジェクトプレフィックスを追加する際に 10,000 オブジェクトの制限を課しています。この制限は、`add_reference` の呼び出しで `max_objects=` を指定することにより調整できます。
{{% /alert %}}

新しい reference artifact `mnist:latest` は、通常のアーティファクトと見た目や動作が似ています。唯一の違いは、アーティファクトが S3/GCS/Azure オブジェクトについてのメタデータ (ETag、サイズ、バージョン ID (バケットでオブジェクトバージョニングが有効な場合) のみで構成されていることです。

W&B は、使用するクラウドプロバイダに基づいたデフォルトのメカニズムを使用して資格情報を探します。使用しているクラウドプロバイダのドキュメントを参照して、使用されている資格情報について詳しく知ってください:

| クラウドプロバイダ | 資格情報ドキュメンテーション |
| ----------------- | ------------------------ |
| AWS            | [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure documentation](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWS の場合、バケットが設定されたユーザーのデフォルトのリージョンにない場合は、`AWS_REGION` 環境変数をバケットのリージョンに設定する必要があります。

このアーティファクトと普通のアーティファクトと同様にやり取りができます。アプリ UI では、ファイルブラウザを使って reference artifact の内容を確認したり、完全な依存関係グラフを探索したり、アーティファクトのバージョン履歴をスキャンすることができます。

{{% alert color="secondary" %}}
画像、音声、ビデオ、ポイントクラウドなどのリッチメディアは、バケットの CORS 設定によってはアプリ UI でレンダリングされないことがあります。バケットの CORS 設定で **app.wandb.ai** を許可リストに追加すると、アプリ UI で適切にリッチメディアをレンダリングできるようになります。

アクセスがプライベートバケットに限定されている場合、アプリ UI でパネルがレンダリングされないことがあります。会社が VPN を持っている場合、バケットのアクセスポリシーを更新し、VPN 内の IP をホワイトリストに追加することができます。
{{% /alert %}}

### Reference Artifact をダウンロードする

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

W&B は、アーティファクトがログされたときに記録されたメタデータを使用して、reference artifact をダウンロードする際に基礎となるバケットからファイルを取得します。バケットでオブジェクトバージョニングが有効な場合、W&B はアーティファクトがログされた時点のファイルの状態に対応するオブジェクトバージョンを取得します。これにより、バケットの内容が進化しても、特定のモデルがトレーニングに使用された正確なデータのイテレーションを指し示すことができます。アーティファクトはトレーニング時のバケットのスナップショットとして機能するためです。

{{% alert %}}
ワークフローの一部としてファイルを上書きする場合は、ストレージバケットで「オブジェクトバージョニング」を有効にすることを W&B は推奨します。バケットでバージョニングが有効な場合、上書きされたファイルへの参照を持つアーティファクトも、古いオブジェクトバージョンを保持するため、依然として無傷です。 

ユースケースに応じて、オブジェクトバージョニングを有効にするための指示を読んでください: [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/using-object-versioning#set)、[Azure](https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-enable)。
{{% /alert %}}

### 全体をまとめる

以下のコード例は、Amazon S3、GCS、または Azure にあるデータセットをトラッキングするためのシンプルなワークフローを示しています。それがトレーニングジョブに投入されます:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")

# アーティファクトをトラッキングし、1度の操作で
# この run の入力としてマークします。バケット内の
# ファイルが変更された場合にのみ、新しいアーティファクト
# バージョンがログされます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# ここでトレーニングを行います...
```

モデルをトラッキングするには、トレーニングスクリプトがバケットにモデルファイルをアップロードした後で、モデルアーティファクトをログします:

```python
import boto3
import wandb

run = wandb.init()

# ここでトレーニング...

s3_client = boto3.client("s3")
s3_client.upload_file("my_model.h5", "my-bucket", "models/cnn/my_model.h5")

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

{{% alert %}}
GCP または Azure の参照によるアーティファクトのトラッキング方法について、エンドツーエンドのウォークスルーを以下のレポートで確認してください:

* [Guide to Tracking Artifacts by Reference](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Working with Reference Artifacts in Microsoft Azure](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}

### ファイルシステムの参照

データセットへの高速なアクセスを実現するためのもう一つの一般的なパターンは、すべてのマシンでトレーニングジョブを実行するリモートファイルシステムに NFS マウントポイントを公開することです。これは、クラウドストレージバケットよりもシンプルなソリューションになることがあります。なぜなら、トレーニングスクリプトの視点から見ると、ファイルはまるでローカルファイルシステム上にあるかのように見えるからです。この使いやすさは、ファイルシステムがマウントされているかどうかに関わらず、Artifacts を使用してファイルシステムへの参照をトラッキングする際にも及びます。

以下の構造を持つファイルシステムが `/mount` にマウントされていると仮定します:

```bash
mount
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` の下にはデータセットである画像のコレクションがあります。これをアーティファクトでトラッキングしましょう:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```

デフォルトでは、W&B はディレクトリへの参照を追加する際に 10,000 ファイルの制限を課しています。この制限は、`add_reference` の呼び出しで `max_objects=` を指定することで調整できます。

URL のトリプルスラッシュに注目してください。最初のコンポーネントは、ファイルシステム参照の使用を示す `file://` プレフィックスです。2番目は、私たちのデータセットへのパスである `/mount/datasets/mnist/` です。

結果として得られるアーティファクト `mnist:latest` は、通常のアーティファクトと同様に見え、動作します。唯一の違いは、アーティファクトがファイルのメタデータ (サイズや MD5 チェックサムなど) のみで構成されており、ファイルそのものはシステムから離れることはない点です。

このアーティファクトは、通常のアーティファクトと同様に操作できます。UI では、ファイルブラウザを使って reference artifact の内容を閲覧したり、完全な依存関係グラフを探索したり、アーティファクトのバージョン履歴をスキャンすることができます。ただし、UI は実際にデータを含んでいないため、画像、音声などのリッチメディアをレンダリングすることはできません。

Reference artifact のダウンロードは簡単です:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

ファイルシステム参照の場合、`download()` 操作は参照されたパスからファイルをコピーして、アーティファクトディレクトリを構築します。上記の例では、`/mount/datasets/mnist` の内容が `artifacts/mnist:v0/` ディレクトリにコピーされます。アーティファクトが上書きされたファイルへの参照を含んでいる場合、`download()` はエラーをスローします。理由は、アーティファクトを再構築できないためです。

すべてをまとめて、マウントされたファイルシステム下にあるデータセットをトラッキングし、トレーニングジョブに供給するシンプルなワークフローを以下に示します:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# アーティファクトをトラッキングし、この run の
# 入力としてマークします。ディレクトリ下のファイルが
# 変更された場合にのみ、新しいアーティファクト
# バージョンがログされます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# ここでトレーニングを行います...
```

モデルをトラッキングするには、トレーニングスクリプトがマウントポイントにモデルファイルを書き込んだ後で、モデルアーティファクトをログします:

```python
import wandb

run = wandb.init()

# ここでトレーニング...

# モデルをディスクに書き込みます

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```