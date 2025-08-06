---
title: 外部ファイルをトラッキングする
description: 外部バケット、HTTPファイルサーバー、または NFS 共有に保存されたファイルをトラッキングします。
menu:
  default:
    identifier: ja-guides-core-artifacts-track-external-files
    parent: artifacts
weight: 7
---

**リファレンスアーティファクト**を使うことで、W&B サーバー外に保存されたファイル（例: CoreWeave AI Object Storage、Amazon S3 バケット、GCS バケット、Azure Blob、HTTP ファイルサーバーや NFS 共有など）の追跡や利用ができます。

W&B はオブジェクトの ETag やサイズなど、オブジェクトに関するメタデータを記録します。バケットでオブジェクトのバージョン管理が有効になっている場合は、そのバージョン ID もログされます。

{{% alert %}}
外部ファイルを参照しないアーティファクトをログする場合、W&B はアーティファクトのファイルを W&B サーバーに保存します。これは、W&B Python SDK でアーティファクトをログする際のデフォルトの動作です。

ファイルやディレクトリを W&B サーバーに保存する方法については、[Artifacts クイックスタート]({{< relref path="/guides/core/artifacts/artifacts-walkthrough" lang="ja" >}}) をご参照ください。
{{% /alert %}}

以下は、リファレンスアーティファクトの構築方法です。

## 外部バケットのアーティファクトを追跡

W&B Python SDK を使って、W&B の外部に保存されたファイルへの参照を追跡します。

1. `wandb.init()` で run を初期化します。
2. `wandb.Artifact()` でアーティファクトオブジェクトを作成します。
3. アーティファクトオブジェクトの `add_reference()` メソッドでバケットパスへの参照を指定します。
4. `run.log_artifact()` でアーティファクトのメタデータをログします。

```python
import wandb

# W&B run を初期化
run = wandb.init()

# アーティファクトオブジェクトを作成
artifact = wandb.Artifact(name="name", type="type")

# バケットパスへの参照を追加
artifact.add_reference(uri = "uri/to/your/bucket/path")

# アーティファクトのメタデータをログ
run.log_artifact(artifact)
run.finish()
```

バケットが次のようなディレクトリ構造の場合を考えます:

```text
s3://my-bucket

|datasets/
  |---- mnist/
|models/
  |---- cnn/
```

`datasets/mnist/` ディレクトリーには画像のコレクションが含まれています。`wandb.Artifact.add_reference()` を使って、このディレクトリー全体をデータセットとして追跡しましょう。以下のサンプルコードは、`add_reference()` メソッドを用いてリファレンスアーティファクト `mnist:latest` を作成します。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact(name="mnist", type="dataset")
artifact.add_reference(uri="s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
run.finish()
```

W&B App では、リファレンスアーティファクトの内容をファイルブラウザで閲覧したり、[依存関係グラフ全体を探索]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}})したり、アーティファクトのバージョン履歴を確認できます。ただし、データ自体はアーティファクトに含まれていないため、画像や音声などのリッチメディアはレンダリングされません。

{{% alert %}}
W&B Artifacts は MinIO を含む、Amazon S3 互換インターフェースに対応しています。下記のスクリプトは、`AWS_S3_ENDPOINT_URL` の環境変数で MinIO サーバーを指定するだけで、そのまま MinIO でも動作します。
{{% /alert %}}

{{% alert color="secondary" %}}
デフォルトでは、W&B はオブジェクトプレフィックスに追加するオブジェクト数の上限を 10,000 に制限しています。この上限は `add_reference()` を呼ぶ際に `max_objects=` を指定することで変更できます。
{{% /alert %}}

## 外部バケットからアーティファクトをダウンロード

W&B はリファレンスアーティファクトをダウンロードする際、記録されたメタデータを元に、基盤となるバケットからファイルを取得します。バケットでオブジェクトのバージョン管理が有効な場合、アーティファクトがログされた時点のファイル状態に対応するオブジェクトバージョンが取得されます。バケットの内容が変化しても、モデルのトレーニング時に使ったデータの正確なバージョンを常に参照することができます。アーティファクトがトレーニング run のバケットのスナップショットとして機能するためです。

以下は、リファレンスアーティファクトをダウンロードする例です。リファレンスアーティファクト・非リファレンスアーティファクトどちらにも同じ API が利用できます。

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

{{% alert %}}
ワークフローの中でファイルを上書きして保存する場合は、ストレージバケットで「オブジェクトバージョン管理」を有効にすることをおすすめします。バージョン管理が有効なバケットに参照されたファイルが上書きされても、過去のバージョンが保持されているため、アーティファクトは影響を受けません。

ご利用環境ごとのオブジェクトバージョン管理の有効化手順については次のリンクをご参照ください: [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/using-object-versioning#set)、[Azure](https://learn.microsoft.com/azure/storage/blobs/versioning-enable)。
{{% /alert %}}

### 外部リファレンスの追加およびダウンロード例

次のコード例は、データセットを Amazon S3 バケットにアップロードし、リファレンスアーティファクトで追跡し、その後ダウンロードする一連の手順です:

```python
import boto3
import wandb

run = wandb.init()

# ここでトレーニング処理...

s3_client = boto3.client("s3")
s3_client.upload_file(file_name="my_model.h5", bucket="my-bucket", object_name="models/cnn/my_model.h5")

# モデルアーティファクトをログ
model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

後で、モデルアーティファクトをダウンロードできます。アーティファクト名とタイプを指定してください:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact(artifact_or_name = "cnn", type="model")
datadir = artifact.download()
```

{{% alert %}}
GCP または Azure を対象に、リファレンス方式でアーティファクトを追跡するエンドツーエンドの解説は下記 Reports をご覧ください:

* [Guide to Tracking Artifacts by Reference with GCP](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Working with Reference Artifacts in Microsoft Azure](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}

## クラウドストレージ認証情報

W&B は、使用しているクラウドプロバイダに従ったデフォルトの方法で認証情報を探します。各クラウドプロバイダで利用される認証情報の詳細は公式ドキュメントをご確認ください。

| クラウドプロバイダ | 認証情報ドキュメント |
| -------------- | ------------------------- |
| CoreWeave AI Object Storage | [CoreWeave AI Object Storage documentation](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/manage-access-keys/cloud-console-tokens) |
| AWS            | [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure documentation](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWS の場合、バケットが設定されているユーザーのデフォルトリージョン以外にある場合は、`AWS_REGION` 環境変数をバケットのリージョンに設定してください。

{{% alert color="secondary" %}}
バケットの CORS 設定によっては、画像・音声・動画・ポイントクラウドのようなリッチメディアが App UI 上で表示できない場合があります。バケットの CORS 設定で **app.wandb.ai** を許可リストに入れておくことで、App UI でリッチメディアが正しく表示されます。

リッチメディア（画像・音声・動画・ポイントクラウドなど）が UI に表示されない場合は、バケットの CORS ポリシーで `app.wandb.ai` が許可されているかご確認ください。
{{% /alert %}}

## ファイルシステム内のアーティファクトを追跡

データセットへの高速アクセスを実現する一般的な方法のひとつに、全ての学習マシンに対してリモートファイルシステムの NFS マウントポイントを提供する方法があります。これはクラウドストレージバケットよりもシンプルな場合があり、トレーニングスクリプトからはローカルファイルシステムと同じような感覚でファイルにアクセスできます。この簡便さは Artifacts でファイルシステムリファレンスを追跡する場合にも変わりません（マウント有無問わず）。

たとえば、`/mount` に次のようなファイルシステムがマウントされているとします:

```bash
mount
|datasets/
		|-- mnist/
|models/
		|-- cnn/
```

`mnist/` ディレクトリには画像データセットが含まれています。これをアーティファクトで追跡するには:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```
{{% alert color="secondary" %}}
デフォルトでは、ディレクトリにリファレンスを追加する際、W&B はファイル数の上限を 10,000 に設定しています。この上限は `add_reference()` で `max_objects=` を指定することで調整できます。
{{% /alert %}}

URL の三重スラッシュにご注目ください。最初の部分はファイルシステムリファレンス用の `file://` プレフィックス、続く部分がデータセットへのパス `/mount/datasets/mnist/` です。

このように作成したアーティファクト `mnist:latest` は、通常のアーティファクト同様に扱えます。唯一の違いはファイルのメタデータ（サイズや MD5 チェックサムなど）のみを保持し、ファイル自体はシステムから移動しない点です。

このアーティファクトも通常のアーティファクトのように操作できます。UI 上ではファイルブラウザからリファレンスアーティファクトの内容を閲覧したり、依存関係グラフ全体を探索したり、アーティファクトのバージョン履歴を確認することができます。ただしデータ自体を保持していないため、画像や音声などのリッチメディアは表示できません。

リファレンスアーティファクトのダウンロード例:

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

ファイルシステムリファレンスの場合、`download()` 操作は参照パスからファイルをコピーしてアーティファクトディレクトリを構成します。上記例では `/mount/datasets/mnist` の内容が `artifacts/mnist:v0/` にコピーされます。アーティファクトが上書きされたファイルのリファレンスを含む場合、`download()` はアーティファクトの再現ができないためエラーになります。

すべてをまとめると、以下のようなコードで、トレーニングジョブで使用するマウントファイルシステム下のデータセットを追跡できます:

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# この run の入力としてアーティファクトを追跡
# ディレクトリ下のファイルが変更された場合のみ新しい
# アーティファクトバージョンが記録されます
run.use_artifact(artifact)

artifact_dir = artifact.download()

# この後でトレーニングなどを実行...
```

モデルを追跡する場合は、トレーニングスクリプトでマウントポイントにモデルファイルを書き込んだ後にモデルアーティファクトをログします:

```python
import wandb

run = wandb.init()

# ここでトレーニング処理...

# モデルをディスクへ書き込む

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```