---
title: Track external files
description: W&B の外部に保存されたファイル (Amazon S3 バケット 、GCS バケット 、HTTP ファイル サーバー 、NFS 共有など)
  を追跡します。
menu:
  default:
    identifier: ja-guides-core-artifacts-track-external-files
    parent: artifacts
weight: 7
---

**参照 Artifacts** を使用して、W&Bシステム外に保存されたファイル（例えば、Amazon S3 バケット、GCS バケット、Azure Blob、HTTP ファイルサーバー、あるいは NFS 共有など）を追跡します。W&B [CLI]({{< relref path="/ref/cli" lang="ja" >}})を使用し、[W&B Run]({{< relref path="/ref/python/run" lang="ja" >}})の外で Artifacts をログ記録します。

### run の外で Artifacts をログ記録する

run の外で Artifact をログ記録すると、W&B は run を作成します。各 Artifact は run に属し、その run は プロジェクト に属します。Artifact（バージョン）はコレクションにも属し、タイプを持ちます。

[`wandb artifact put`]({{< relref path="/ref/cli/wandb-artifact/wandb-artifact-put" lang="ja" >}}) コマンドを使用すると、W&B run の外で Artifact を W&B サーバーにアップロードできます。Artifact が属する プロジェクト の名前と Artifact の名前（`project/artifact_name`）を指定します。オプションで、タイプ（`TYPE`）を指定します。以下のコードスニペットの `PATH` を、アップロードする Artifact のファイルパスに置き換えます。

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

指定した プロジェクト が存在しない場合、W&B は新しい プロジェクト を作成します。Artifact のダウンロード方法については、[Artifacts のダウンロードと使用]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}})を参照してください。

## W&B の外で Artifacts を追跡する

W&B Artifacts を使用して、データセット の バージョン管理 と モデルリネージ を行い、**参照 Artifacts** を使用して W&B サーバーの外に保存されたファイルを追跡します。このモードでは、Artifact は URL、サイズ、チェックサムなどのファイルに関する メタデータ のみを保存します。基になる データ がシステムから離れることはありません。ファイルを W&B サーバーに保存する方法については、[クイックスタート]({{< relref path="/guides/core/artifacts/artifacts-walkthrough" lang="ja" >}})を参照してください。

以下では、参照 Artifacts を構築する方法と、ワークフロー に組み込む最適な方法について説明します。

### Amazon S3 / GCS / Azure Blob Storage 参照

W&B Artifacts を使用して、データセット と モデル の バージョン管理 を行い、クラウド ストレージ バケット内の参照を追跡します。Artifact の参照を使用すると、既存のストレージ レイアウトを変更することなく、シームレスに追跡を バケット の上に重ねることができます。

Artifacts は、基盤となるクラウド ストレージ ベンダー（AWS、GCP、Azure など）を抽象化します。以下のセクションで説明する情報は、Amazon S3、Google Cloud Storage、Azure Blob Storage に均一に適用されます。

{{% alert %}}
W&B Artifacts は、MinIO を含む、Amazon S3 互換のインターフェースをサポートします。`AWS_S3_ENDPOINT_URL` 環境変数 を MinIO サーバーを指すように設定すると、以下の スクリプト はそのまま動作します。
{{% /alert %}}

次の構造の バケット があると仮定します。

```bash
s3://my-bucket
+-- datasets/
| +-- mnist/
+-- models/
 +-- cnn/
```

`mnist/` の下には、画像のコレクションである データセット があります。これを Artifact で追跡してみましょう。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```
{{% alert color="secondary" %}}
デフォルトでは、W&B はオブジェクト プレフィックス を追加する際に 10,000 オブジェクト の制限を課します。この制限は、`add_reference` の呼び出しで `max_objects=` を指定することで調整できます。
{{% /alert %}}

新しい参照 Artifact `mnist:latest` は、通常の Artifact と同様に見え、動作します。唯一の違いは、Artifact が S3/GCS/Azure オブジェクト に関する メタデータ （ETag、サイズ、オブジェクト の バージョン管理 が バケット で有効になっている場合は バージョン ID など）のみで構成されていることです。

W&B は、使用している クラウド プロバイダー に基づいて、認証情報 を検索するためのデフォルトのメカニズムを使用します。使用する認証情報 の詳細については、 クラウド プロバイダー の ドキュメント を参照してください。

| クラウド プロバイダー | 認証情報 の ドキュメント |
| -------------- | ------------------------- |
| AWS | [Boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#configuring-credentials) |
| GCP | [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) |
| Azure | [Azure documentation](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) |

AWS の場合、 バケット が設定された ユーザー のデフォルト リージョン にない場合は、`AWS_REGION` 環境変数 を バケット リージョン と一致するように設定する必要があります。

この Artifact は、通常の Artifact と同様に操作します。App UI では、ファイル ブラウザー を使用して参照 Artifact のコンテンツ を調べたり、完全な依存関係グラフ を調べたり、Artifact の バージョン管理 された履歴をスキャンしたりできます。

{{% alert color="secondary" %}}
画像、オーディオ、ビデオ、ポイント クラウド などのリッチ メディア は、 バケット の CORS 設定によっては、App UI で レンダリング されない場合があります。 バケット の CORS 設定で **app.wandb.ai** のリストを許可すると、App UI でそのようなリッチ メディア を適切に レンダリング できます。

会社の VPN がある場合、プライベート バケット の パネル が App UI で レンダリング されない場合があります。VPN 内の IP を ホワイトリスト に登録するように バケット の アクセス ポリシー を更新できます。
{{% /alert %}}

### 参照 Artifact をダウンロードする

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

W&B は、参照 Artifact をダウンロードする際に、Artifact がログ記録されたときに記録された メタデータ を使用して、基盤となる バケット からファイル を取得します。 バケット で オブジェクト の バージョン管理 が有効になっている場合、W&B は Artifact がログ記録された時点でのファイルの 状態 に対応する オブジェクト バージョン を取得します。これは、 バケット のコンテンツ を進化させても、特定の モデル がトレーニング された データの 正確な イテレーション を指すことができることを意味します。Artifact は トレーニング 時の バケット の スナップショット として機能するためです。

{{% alert %}}
ワークフロー の一部としてファイルを上書きする場合は、ストレージ バケット で「オブジェクト の バージョン管理 」を有効にすることをお勧めします。 バケット で バージョン管理 が有効になっている場合、上書きされたファイルへの参照を持つ Artifacts は、古い オブジェクト バージョン が保持されるため、そのまま残ります。

ユースケース に基づいて、オブジェクト の バージョン管理 を有効にする 手順 を読んでください：[AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html), [GCP](https://cloud.google.com/storage/docs/using-object-versioning#set), [Azure](https://learn.microsoft.com/en-us/azure/storage/blobs/versioning-enable).
{{% /alert %}}

### 統合する

次のコード例は、Amazon S3、GCS、または Azure で データセット を追跡するために使用できる簡単な ワークフロー を示しています。

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")

# Artifact を追跡し、この run への入力としてマークします。
# バケット 内のファイル が変更された場合にのみ、新しい Artifact バージョン がログ記録されます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# ここで トレーニング を実行します...
```

モデル を追跡するために、 トレーニング スクリプト が モデル ファイル を バケット にアップロードした後、 モデル Artifact をログ記録できます。

```python
import boto3
import wandb

run = wandb.init()

# トレーニング はこちら...

s3_client = boto3.client("s3")
s3_client.upload_file("my_model.h5", "my-bucket", "models/cnn/my_model.h5")

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("s3://my-bucket/models/cnn/")
run.log_artifact(model_artifact)
```

{{% alert %}}
GCP または Azure の参照によって Artifacts を追跡する方法のエンドツーエンド の チュートリアル については、次の レポート を参照してください。

* [参照による Artifacts 追跡 ガイド](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)
* [Microsoft Azure での参照 Artifacts の操作](https://wandb.ai/andrea0/azure-2023/reports/Efficiently-Harnessing-Microsoft-Azure-Blob-Storage-with-Weights-Biases--Vmlldzo0NDA2NDgw)
{{% /alert %}}

### ファイルシステム 参照

データセット に 高速 に アクセス するためのもう 1 つの一般的なパターンは、すべての トレーニング ジョブ を実行しているマシンでリモート ファイルシステム に NFS マウント ポイント を公開することです。トレーニング スクリプト の観点から見ると、ファイル がローカル ファイルシステム にあるように見えるため、これは クラウド ストレージ バケット よりもさらに簡単なソリューションになる可能性があります。幸いなことに、その使いやすさは、ファイルシステム への参照を追跡するために Artifacts を使用することにも拡張されます（マウントされているかどうかに関係なく）。

次の構造のファイルシステム が `/mount` にマウントされていると仮定します。

```bash
mount
+-- datasets/
| +-- mnist/
+-- models/
 +-- cnn/
```

`mnist/` の下には、画像のコレクションである データセット があります。これを Artifact で追跡してみましょう。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")
run.log_artifact(artifact)
```

デフォルトでは、W&B は ディレクトリー への参照を追加する際に 10,000 ファイル の制限を課します。この制限は、`add_reference` の呼び出しで `max_objects=` を指定することで調整できます。

URL には 3 つのスラッシュ があることに注意してください。最初のコンポーネント は、ファイルシステム 参照の使用を示す `file://` プレフィックス です。2 番目のコンポーネント は、 データセット へのパス `/mount/datasets/mnist/` です。

結果として得られる Artifact `mnist:latest` は、通常の Artifact と同じように見え、動作します。唯一の違いは、Artifact がファイル に関する メタデータ （サイズ や MD5 チェックサム など）のみで構成されていることです。ファイル 自体がシステムから離れることはありません。

この Artifact は、通常の Artifact と同じように操作できます。UI では、ファイル ブラウザー を使用して参照 Artifact のコンテンツ を参照したり、完全な依存関係グラフ を調べたり、Artifact の バージョン管理 された履歴をスキャンしたりできます。ただし、 データ 自体が Artifact に含まれていないため、UI は画像 やオーディオ などのリッチ メディア を レンダリング できません。

参照 Artifact のダウンロードは簡単です。

```python
import wandb

run = wandb.init()
artifact = run.use_artifact("entity/project/mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

ファイルシステム 参照の場合、`download()` 操作は参照されたパスからファイルをコピーして Artifact ディレクトリー を構築します。上記の例では、`/mount/datasets/mnist` のコンテンツ が ディレクトリー `artifacts/mnist:v0/` にコピーされます。Artifact に上書きされたファイル への参照が含まれている場合、Artifact を再構築できなくなるため、`download()` は エラー をスローします。

すべてをまとめると、マウントされたファイルシステム の下にある データセット を追跡するために使用できる簡単な ワークフロー を次に示します。

```python
import wandb

run = wandb.init()

artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("file:///mount/datasets/mnist/")

# Artifact を追跡し、この run への入力としてマークします。
# ディレクトリー の下のファイル が変更された場合にのみ、新しい Artifact バージョン がログ記録されます。
run.use_artifact(artifact)

artifact_dir = artifact.download()

# ここで トレーニング を実行します...
```

モデル を追跡するために、 トレーニング スクリプト が モデル ファイル をマウント ポイント に書き込んだ後、 モデル Artifact をログ記録できます。

```python
import wandb

run = wandb.init()

# トレーニング はこちら...

# モデル を ディスク に書き込みます

model_artifact = wandb.Artifact("cnn", type="model")
model_artifact.add_reference("file:///mount/cnn/my_model.h5")
run.log_artifact(model_artifact)
```