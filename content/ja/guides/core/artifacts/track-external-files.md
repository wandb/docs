---
title: 外部ファイルをトラックする
description: W&B の外部に保存されたファイルも、Amazon S3 バケット、GCS バケット、HTTP ファイルサーバー、または NFS 共有内のファイルとしてトラックできます。
menu:
  default:
    identifier: ja-guides-core-artifacts-track-external-files
    parent: artifacts
weight: 7
---

**リファレンスアーティファクト**を使用して、Amazon S3 バケット、GCS バケット、Azure blob、HTTP ファイルサーバー、または NFS シェアなど、W&B システムの外部に保存されたファイルをトラッキングします。 W&B [CLI]({{< relref path="/ref/cli" lang="ja" >}})を使用して、[W&B Run]({{< relref path="/ref/python/run" lang="ja" >}})の外部でアーティファクトをログします。

### Run の外部でアーティファクトをログする

W&B は、run の外部でアーティファクトをログするときに run を作成します。各アーティファクトは run に属し、run はプロジェクトに属します。アーティファクト (バージョン) もコレクションに属し、タイプを持ちます。

[`wandb artifact put`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put" lang="ja" >}}) コマンドを使用して、W&B の run の外部でアーティファクトを W&B サーバーにアップロードします。アーティファクトを属させたいプロジェクトの名前とアーティファクトの名前 (`project/artifact_name`) を指定します。必要に応じて、タイプ (`TYPE`) を指定します。以下のコードスニペットでは、アップロードしたいアーティファクトのファイルパスに `PATH` を置き換えてください。

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

指定したプロジェクトが存在しない場合、W&B は新しいプロジェクトを作成します。アーティファクトのダウンロード方法については、[アーティファクトのダウンロードと使用]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}})を参照してください。

## W&B の外部でアーティファクトをトラッキングする

W&B Artifacts をデータセットのバージョン管理やモデルのリネージに使用し、**リファレンスアーティファクト**を使用して W&B サーバーの外部に保存されたファイルをトラッキングします。このモードでは、アーティファクトはファイルに関するメタデータ (例えば、URL、サイズ、チェックサム) のみを保存します。基礎データはシステムから離れることはありません。ファイルとディレクトリーを W&B サーバーに保存する方法については、[クイックスタート]({{< relref path="/guides/core/artifacts/artifacts-walkthrough" lang="ja" >}})を参照してください。

以下は、リファレンスアーティファクトを作成し、それをワークフローに最適に組み込む方法を説明します。

### Amazon S3 / GCS / Azure Blob Storage リファレンス

W&B Artifacts をデータセットとモデルのバージョン管理に使用して、クラウドストレージバケットでのリファレンスをトラッキングします。アーティファクトリファレンスを使用すると、既存のストレージレイアウトに変更を加えることなく、バケットの上にシームレスにトラッキングをレイヤリングできます。

Artifacts は基礎となるクラウドストレージベンダー (AWS、GCP、Azure など) を抽象化します。次のセクションで説明される情報は、Amazon S3、Google Cloud Storage、Azure Blob Storage に共通して適用されます。

{{% alert %}}
W&B Artifacts は、MinIO を含む任意の Amazon S3 互換インターフェースをサポートしています。 `AWS_S3_ENDPOINT_URL` 環境変数を MinIO サーバーを指すように設定すれば、以下のスクリプトはそのまま動作します。
{{% /alert %}}

次の構造を持つバケットがあると仮定します：

```bash
s3://my-bucket
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` の下には、私たちのデータセットである画像のコレクションがあります。アーティファクトでそれをトラッキングしましょう：

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

{{% alert color="secondary" %}}
デフォルトでは、W&B はオブジェクトプリフィックスを追加する際に 10,000 オブジェクトの制限を課しています。この制限は、`add_reference` の呼び出しで `max_objects=` を指定することによって調整できます。
{{% /alert %}}

新しいリファレンスアーティファクト `mnist:latest` は、通常のアーティファクトと非常に似た外観と挙動を持っています。唯一の違いは、アーティファクトが S3/GCS/Azure オブジェクトに関するメタデータ (例えば、ETag、サイズ、バージョン ID) のみを含んでいることです (バケットにオブジェクトのバージョン管理が有効になっている場合)。

W&B は、使用するクラウドプロバイダーに基づいてクレデンシャルを探すデフォルトのメカニズムを使用します。クラウドプロバイダーからのドキュメントを読み、使用されるクレデンシャルについて詳しく学びましょう。

| クラウドプロバイダー | クレデンシャルドキュメント |
| -------------- | ------------------------- |
| AWS            | [Boto3 ドキュメント](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP            | [Google Cloud ドキュメント](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure          | [Azure ドキュメント](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWS では、バケットが設定されたユーザーのデフォルトリージョンに位置していない場合、`AWS_REGION` 環境変数をバケットリージョンに一致させる必要があります。

このアーティファクトを通常のアーティファクトのように扱うことができます。アプリ UI では、ファイルブラウザを使用してリファレンスアーティファクトの内容を閲覧したり、完全な依存関係グラフを探索したり、アーティファクトのバージョン履歴をスキャンしたりできます。

{{% alert color="secondary" %}}
画像、オーディオ、ビデオ、ポイントクラウドといったリッチメディアは、バケットの CORS 設定によってアプリ UI で適切にレンダリングされない可能性があります。バケットの CORS 設定で **app.wandb.ai** を許可リストに追加することで、アプリ UI でこれらのリッチメディアが正しくレンダリングされるようになります。

パネルは、プライベートバケットの場合アプリ UI でレンダリングされないかもしれません。もし会社が VPN を使用している場合は、VPN 内の IP をホワイトリストに追加するようにバケットのアクセスポリシーを更新できます。
{{% /alert %}}

### リファレンスアーティファクトをダウンロードする

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

W&B は、リファレンスアーティファクトをダウンロードする際に、アーティファクトがログされたときに記録されたメタデータを使用して、基となるバケットからファイルを取得します。バケットにオブジェクトのバージョン管理が有効になっている場合、W&B はアーティファクトがログされた時点のファイルの状態に対応するオブジェクトバージョンを取得します。これは、バケットの内容が進化しても、アーティファクトがトレーニングされた特定の反復にあなたのデータを指し示すことができることを意味します。アーティファクトはトレーニング時点でのバケットのスナップショットとして機能します。

{{% alert %}}
ワークフローの一環としてファイルを上書きする場合は、W&B はストレージバケットの「オブジェクトバージョン管理」を有効にすることを推奨します。バケットにバージョン管理が有効になっている場合、上書きされたファイルへのリファレンスを持つアーティファクトも依然として無傷であることになります。なぜなら、古いバージョンのオブジェクトが保持されるからです。

ユースケースに基づいて、オブジェクトバージョン管理を有効にする手順をお読みください。[AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)、[GCP](https://cloud.google.com/storage/docs/using-object-versioning#set)、[Azure](https://learn.microsoft.com/azure/storage/blobs/versioning-enable)。
{{% /alert %}}

### すべてを結び付ける

次のコード例は、トレーニングジョブに供給される Amazon S3、GCS、または Azure 上のデータセットをトラッキングするために使用できる単純なワークフローを示しています：

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")

# アーティファクトをトラッキングし、それを
# この run の入力としてマークします。
# バケット内のファイルが変更された場合にのみ、新しい
# アーティファクトバージョンがログされます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# トレーニングをここで実行...
```

モデルをトラッキングするために、トレーニングスクリプトがモデルファイルをバケットにアップロードした後に、モデルアーティファクトをログできます：

```python
import boto3
import wandb

run = wandb.init()

# トレーニングをここで実行...

s3_client = boto3.client("s3")
s3_client.upload_file("my_model.h5", "my-bucket", "models/cnn/my_model.h5")

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

{{% alert %}}
GCP または Azure のリファレンスでのアーティファクトのトラッキング方法についてのエンドツーエンドのガイドを読むには、次のレポートをご覧ください：

* [リファレンスでのアーティファクトトラッキングガイド](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Microsoft Azure でのリファレンスアーティファクトの作業](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}

### ファイルシステムリファレンス

データセットへの高速アクセスのためのもう一つの一般的なパターンは、NFS マウントポイントをトレーニングジョブを実行するすべてのマシンでリモートファイルシステムに公開することです。これは、クラウドストレージバケットよりもさらに簡単なソリューションになる可能性があります。トレーニングスクリプトの視点からは、ファイルはちょうどローカルファイルシステムに置かれているかのように見えるからです。幸運にも、その使いやすさは、ファイルシステムへのリファレンスをトラッキングするために Artifacts を使用する場合にも当てはまります。ファイルシステムがマウントされているかどうかに関係なくです。

次の構造を持つファイルシステムが `/mount` にマウントされていると仮定します：

```bash
mount
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```

`mnist/` の下には、私たちのデータセットである画像のコレクションがあります。アーティファクトでそれをトラッキングしましょう：

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```

デフォルトでは、W&B はディレクトリへのリファレンスを追加する際に 10,000 ファイルの制限を課しています。この制限は、`add_reference` の呼び出しで `max_objects=` を指定することによって調整できます。

URL のトリプルスラッシュに注目してください。最初のコンポーネントは、ファイルシステムリファレンスの使用を示す `file://` プレフィックスです。二番目は、データセットのパス `/mount/datasets/mnist/` です。

結果として得られるアーティファクト `mnist:latest` は通常のアーティファクトのように見え、機能します。唯一の違いは、アーティファクトがファイルに関するメタデータ (サイズや MD5 チェックサムなど) のみを含んでいることです。ファイル自体はシステムから離れることはありません。

このアーティファクトを通常のアーティファクトのように操作できます。UI では、ファイルブラウザを使用してリファレンスアーティファクトの内容を閲覧したり、完全な依存関係グラフを探索したり、アーティファクトのバージョン履歴をスキャンしたりできます。ただし、アーティファクト自体にデータが含まれていないため、UI では画像、オーディオなどのリッチメディアをレンダリングできません。

リファレンスアーティファクトをダウンロードするのは簡単です：

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

ファイルシステムリファレンスの場合、`download()` 操作は参照されたパスからファイルをコピーして、アーティファクトディレクトリを構築します。上記の例では、`/mount/datasets/mnist` の内容がディレクトリ `artifacts/mnist:v0/` にコピーされます。アーティファクトが上書きされたファイルへのリファレンスを含む場合、`download()` はエラーを投げます。アーティファクトがもはや再構築できないからです。

すべてをまとめると、ここにマウントされたファイルシステムの下のデータセットをトラッキングして、トレーニングジョブに供給するために使用できる簡単なワークフローがあります：

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# アーティファクトをトラッキングし、それを
# この run の入力としてマークします。ディレクトリ下の
# ファイルが変更された場合にのみ、新しいアーティファクト
# バージョンがログされます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# トレーニングをここで実行...
```

モデルをトラッキングするために、トレーニングスクリプトがモデルファイルをマウントポイントに書き込んだ後に、モデルアーティファクトをログできます：

```python
import wandb

run = wandb.init()

# トレーニングをここで実行...

# モデルをディスクに書き込む

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```