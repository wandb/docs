---
title: Track external files
description: W&B の外部 (Amazon S3 バケット 、GCS バケット 、HTTP ファイル サーバー 、NFS 共有など) に保存されたファイルを追跡します。
menu:
  default:
    identifier: ja-guides-core-artifacts-track-external-files
    parent: artifacts
weight: 7
---

**reference artifacts** を使用すると、たとえば Amazon S3 バケット、GCS バケット、Azure Blob、HTTP ファイルサーバー、NFS 共有など、W&B システムの外部に保存されたファイルを追跡できます。W&B [CLI]({{< relref path="/ref/cli" lang="ja" >}}) を使用して、[W&B Run]({{< relref path="/ref/python/run" lang="ja" >}}) の外部で Artifacts を記録します。

### run の外部で Artifacts を記録する

W&B は、run の外部で Artifacts を記録すると run を作成します。各 artifact は run に属し、run はプロジェクトに属します。artifact (バージョン) はコレクションにも属し、タイプを持ちます。

[`wandb artifact put`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put" lang="ja" >}}) コマンドを使用すると、W&B run の外部で Artifacts を W&B サーバーにアップロードできます。artifact を所属させるプロジェクトの名前と artifact の名前 (`project/artifact_name`) を指定します。オプションでタイプ (`TYPE`) を指定します。以下のコードスニペットの `PATH` を、アップロードする artifact のファイルパスに置き換えます。

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

指定したプロジェクトが存在しない場合、W&B は新しいプロジェクトを作成します。artifact のダウンロード方法については、[Artifacts のダウンロードと使用]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}}) を参照してください。

## W&B の外部で Artifacts を追跡する

データセットの バージョン管理 と モデルリネージ に W&B Artifacts を使用し、**reference artifacts** を使用して W&B サーバーの外部に保存されたファイルを追跡します。このモードでは、artifact は URL、サイズ、チェックサムなどのファイルに関する メタデータ のみを保存します。基になるデータがシステムから離れることはありません。代わりにファイルと ディレクトリー を W&B サーバーに保存する方法については、[クイックスタート]({{< relref path="/guides/core/artifacts/artifacts-walkthrough" lang="ja" >}}) を参照してください。

以下では、reference artifacts の構築方法と、ワークフローに組み込む最適な方法について説明します。

### Amazon S3 / GCS / Azure Blob Storage の参照

データセットとモデルの バージョン管理 に W&B Artifacts を使用して、クラウドストレージ バケット 内の参照を追跡します。artifact の参照を使用すると、既存のストレージレイアウトを変更することなく、バケットの上にシームレスに追跡を重ねることができます。

Artifacts は、基盤となるクラウドストレージ ベンダー (AWS、GCP、Azure など) を抽象化します。以下のセクションで説明する情報は、Amazon S3、Google Cloud Storage、Azure Blob Storage に一様に適用されます。

{{% alert %}}
W&B Artifacts は、MinIO を含む Amazon S3 互換インターフェースをサポートしています。`AWS_S3_ENDPOINT_URL` 環境変数を MinIO サーバーを指すように設定すると、以下のスクリプトはそのまま動作します。
{{% /alert %}}

次の構造のバケットがあると仮定します。

```bash
s3://my-bucket
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` の下には、画像のコレクションであるデータセットがあります。これを artifact で追跡してみましょう。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```
{{% alert color="secondary" %}}
デフォルトでは、W&B は オブジェクト のプレフィックスを追加する際に 10,000 オブジェクト の制限を課します。`add_reference` の呼び出しで `max_objects=` を指定すると、この制限を調整できます。
{{% /alert %}}

新しい reference artifact `mnist:latest` は、通常の artifact と同様に見え、動作します。唯一の違いは、artifact が S3/GCS/Azure オブジェクト に関する メタデータ (ETag、サイズ、バージョン ID (オブジェクト の バージョン管理 がバケット で有効になっている場合)) のみで構成されていることです。

W&B は、使用するクラウドプロバイダーに基づいて、認証情報を検索するためのデフォルトのメカニズムを使用します。使用される認証情報の詳細については、クラウドプロバイダーのドキュメントをお読みください。

| クラウドプロバイダー | 認証情報のドキュメント |
| -------------- | ------------------------- |
| AWS            | [Boto3 のドキュメント](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud のドキュメント](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure のドキュメント](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWS の場合、バケットが設定された ユーザー のデフォルトリージョンにない場合は、`AWS_REGION` 環境変数をバケットリージョンと一致するように設定する必要があります。

この artifact は、通常の artifact と同様に操作します。App UI では、ファイルブラウザーを使用して reference artifact の内容を確認したり、完全な依存関係グラフを調べたり、artifact の バージョン管理 された履歴をスキャンしたりできます。

{{% alert color="secondary" %}}
イメージ、オーディオ、ビデオ、点群などのリッチメディアは、バケット の CORS 設定によっては、App UI でレンダリングされない場合があります。バケット の CORS 設定で **app.wandb.ai** を許可リストに登録すると、App UI でこのようなリッチメディアを適切にレンダリングできるようになります。

プライベート バケット の場合、 パネル が App UI でレンダリングされない可能性があります。会社に VPN がある場合は、VPN 内の IP をホワイトリストに登録するようにバケット の アクセス ポリシーを更新できます。
{{% /alert %}}

### reference artifact をダウンロードする

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

W&B は、reference artifact をダウンロードするときに、artifact が記録されたときに記録された メタデータ を使用して、基になるバケット からファイルを取得します。バケット で オブジェクト の バージョン管理 が有効になっている場合、W&B は artifact が記録されたときのファイルの状態に対応する オブジェクト バージョンを取得します。つまり、バケット の内容を進化させても、artifact は トレーニング 時のバケット の スナップショット として機能するため、特定のモデルが トレーニング されたデータの正確な イテレーション を指すことができます。

{{% alert %}}
ワークフロー の一部としてファイルを上書きする場合は、ストレージ バケット で「 オブジェクト の バージョン管理 」を有効にすることをお勧めします。バケット で バージョン管理 を有効にすると、上書きされたファイルへの参照を含む Artifacts は、古い オブジェクト バージョンが保持されるため、引き続きそのまま残ります。

ユースケース に基づいて、オブジェクト の バージョン管理 を有効にする手順をお読みください: [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/using-object-versioning#set), [Azure](https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-enable)。
{{% /alert %}}

### 統合する

次のコード例は、Amazon S3、GCS、または Azure でデータセットを追跡するために使用できる簡単な ワークフロー を示しており、 トレーニング ジョブにフィードされます。

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")

# Artifact を追跡し、これを入力としてマークします
# この run で一度に実行します。新しい artifact バージョン
# バケット 内のファイルが変更された場合にのみ記録されます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# ここで トレーニング を実行します...
```

モデルを追跡するために、 トレーニング スクリプトがモデルファイルをバケット にアップロードした後、モデル artifact を記録できます。

```python
import boto3
import wandb

run = wandb.init()

# ここで トレーニング を実行します...

s3_client = boto3.client("s3")
s3_client.upload_file("my_model.h5", "my-bucket", "models/cnn/my_model.h5")

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

{{% alert %}}
GCP または Azure の参照によって Artifacts を追跡する方法のエンドツーエンド の チュートリアル については、次の Reports をお読みください。

* [参照による Artifacts の追跡 ガイド](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Microsoft Azure での Reference Artifacts の操作](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}

### ファイルシステムの参照

データセットへの高速 アクセス のためのもう 1 つの一般的な パターン は、すべてのマシンで トレーニング ジョブを実行しているリモートファイルシステムに NFS マウントポイントを公開することです。これは、 トレーニング スクリプトの観点から見ると、ファイルがローカルファイルシステムに配置されているように見えるため、クラウドストレージ バケット よりもさらに簡単なソリューションになる可能性があります。幸いなことに、その使いやすさは、ファイルシステム (マウントされているかどうか) への参照を追跡するために Artifacts を使用することにも拡張されます。

次の構造で `/mount` にマウントされたファイルシステムがあると仮定します。

```bash
mount
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` の下には、画像のコレクションであるデータセットがあります。artifact で追跡してみましょう。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```

デフォルトでは、W&B は ディレクトリー への参照を追加する際に 10,000 ファイルの制限を課します。`add_reference` の呼び出しで `max_objects=` を指定すると、この制限を調整できます。

URL の 3 つのスラッシュに注意してください。最初のコンポーネントは、ファイルシステムの参照の使用を示す `file://` プレフィックスです。2 番目のコンポーネントは、データセットへのパス `/mount/datasets/mnist/` です。

結果として得られる artifact `mnist:latest` は、通常の artifact と同様に見え、動作します。唯一の違いは、artifact がファイルに関する メタデータ (サイズや MD5 チェックサムなど) のみで構成されていることです。ファイル自体はシステムから離れることはありません。

この artifact は、通常の artifact と同様に操作できます。UI では、ファイルブラウザーを使用して reference artifact の内容を参照したり、完全な依存関係グラフを調べたり、artifact の バージョン管理 された履歴をスキャンしたりできます。ただし、データ自体が artifact に含まれていないため、UI は画像、オーディオなどのリッチメディアをレンダリングできません。

reference artifact のダウンロードは簡単です。

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

ファイルシステムの参照の場合、`download()` 操作は、参照されたパスからファイルをコピーして、artifact ディレクトリー を構築します。上記の例では、`/mount/datasets/mnist` の内容が ディレクトリー `artifacts/mnist:v0/` にコピーされます。artifact に上書きされたファイルへの参照が含まれている場合、artifact を再構築できなくなるため、`download()` はエラーをスローします。

すべてをまとめると、マウントされたファイルシステムの下でデータセットを追跡するために使用できる簡単な ワークフロー を次に示します。

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# Artifact を追跡し、これを入力としてマークします
# この run で一度に実行します。新しい artifact バージョン
# ディレクトリー の下のファイルが
# 変更されました。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# ここで トレーニング を実行します...
```

モデルを追跡するために、 トレーニング スクリプトがモデルファイルをマウントポイントに書き込んだ後、モデル artifact を記録できます。

```python
import wandb

run = wandb.init()

# ここで トレーニング を実行します...

# モデルをディスクに書き込む

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```
