---
description: >-
  Track files saved outside the Weights & Biases such as in an Amazon S3 bucket,
  GCS bucket, HTTP file server, or even an NFS share.
displayed_sidebar: ja
---

# 外部ファイルのトラッキング

<head>
	<title>参照アーティファクトを使って外部ファイルをトラッキングする</title>
</head>
**リファレンスアーティファクト**を使用して、Weights & Biasesシステムの外部に保存されたファイル（例：Amazon S3バケット、GCSバケット、HTTPファイルサーバー、あるいはNFS共有）をトラッキングします。 W&B [CLI](https://docs.wandb.ai/ref/cli) を使用して、[W&B Run](https://docs.wandb.ai/ref/python/run) の外部でアーティファクトをログに記録します。

### Runの外部でアーティファクトをログに記録する

Runの外部でアーティファクトをログに記録すると、Weights & BiasesはRunを作成します。各アーティファクトはRunに属し、Runはプロジェクトに属します。また、アーティファクト（バージョン）はコレクションにも属しており、タイプがあります。

W&B Runの外部でアーティファクトをW&Bサーバーにアップロードするには、[`wandb artifact put`](https://docs.wandb.ai/ref/cli/wandb-artifact/wandb-artifact-put)コマンドを使用します。 アーティファクトが所属するプロジェクト名とアーティファクト名（`project/artifact_name`）を指定してください。また、オプションでタイプ（`TYPE`）を指定します。以下のコードスニペットの`PATH`を、アップロードしたいアーティファクトのファイルパスに置き換えてください。

```bash
$ wandb artifact put --name project/artifact_name --type TYPE PATH
```

Weights & Biasesは、指定したプロジェクトが存在しない場合、新しいプロジェクトを作成します。アーティファクトのダウンロード方法については、[アーティファクトのダウンロードと使用](https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact)を参照してください。
## Weights & Biasesを使わずにアーティファクトをトラッキングする

Weights & Biases Artifactsをデータセットのバージョン管理とモデルの履歴用に使い、W&Bサーバーの外に保存されたファイルをトラッキングするために**リファレンスアーティファクト**を使用します。このモードでは、アーティファクトはファイルのメタデータのみを保存し、URL、サイズ、チェックサムなどが含まれます。実際のデータはシステムから離れることはありません。ファイルやディレクトリをW&Bサーバーに代わりに保存する方法については、[クイックスタート](https://docs.wandb.ai/guides/artifacts/quickstart)をご覧ください。

GCPでリファレンスファイルをトラッキングする例については、[リファレンスでアーティファクトをトラッキングするガイド](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)を参照してください。

以下では、リファレンスアーティファクトの構築方法やワークフローに最適な方法で組み込む方法について説明します。
### Amazon S3 / GCS 参照

ウェイト＆バイアスのアーティファクトを使って、データセットとモデルのバージョン管理を行い、クラウドストレージのバケット内の参照を追跡します。アーティファクト参照を使用することで、既存のストレージレイアウトを変更することなく、バケットの上に追跡をシームレスに追加できます。

アーティファクトは、クラウドストレージのベンダー（AWSやGCPなど）に基づいて抽象化されます。前のセクションで説明された情報は、Google Cloud StorageとAmazon S3の両方に同様に適用されます。

:::info
Weights & Biasesのアーティファクトは、Amazon S3と互換性のあるインターフェース（MinIOを含む）をサポートしています！下記のスクリプトは、AWS\_S3\_ENDPOINT\_URL環境変数をMinIOサーバーを指すように設定することでそのまま動作します。
:::
以下の構造を持つバケットがあるとします：

```
s3://my-bucket
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```
`mnist/`の下には、画像のコレクションであるデータセットがあります。それをアーティファクトでトラッキングしましょう。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact('mnist', type='dataset')
artifact.add_reference('s3://my-bucket/datasets/mnist')
run.log_artifact(artifact)
```

:::caution
デフォルトでは、W&Bはオブジェクトプレフィックスの追加時に10,000オブジェクトの制限を設けています。この制限を調整するには、`add_reference`への呼び出しで`max_objects=`を指定します。
:::
新しいリファレンスアーティファクト `mnist:latest` は、通常のアーティファクトと同様の見た目と振る舞いをします。唯一の違いは、このアーティファクトがS3/GCSオブジェクトに関するメタデータ（ETag、サイズ、バージョンID（バケットでオブジェクトバージョン管理が有効になっている場合））のみで構成されていることです。

Weights & Biasesは、Amazon S3またはGCSバケットへの参照を追加する際に、クラウドプロバイダーに関連付けられた対応するクレデンシャルファイルや環境変数を使用しようとします。

| 優先順位                    | Amazon S3                                                                                                           | Google Cloud Storage                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1 - 環境変数                 | <p><code>AWS_ACCESS_KEY_ID</code></p><p><code>AWS_SECRET_ACCESS_KEY</code></p><p><code>AWS_SESSION_TOKEN</code></p> | `GOOGLE_APPLICATION_CREDENTIALS`                              |
| 2 - 共有クレデンシャルファイル | `~/.aws/credentials`                                                                                                | `application_default_credentials.json` in `~/.config/gcloud/` |
| 3 - 設定ファイル             | `~/.aws.config`                                                                                                     | N/A                                                           |
通常のアーティファクトと同様に、このアーティファクトともやりとりができます。アプリUIでは、ファイルブラウザを使って参照アーティファクトの内容を確認したり、完全な依存関係グラフを調べたり、アーティファクトのバージョン管理された履歴を確認することができます。

:::caution
画像、オーディオ、ビデオ、ポイントクラウドなどのリッチメディアは、バケットのCORS設定によっては、アプリUIで表示できないことがあります。**app.wandb.ai** をバケットのCORS設定にリスト化することで、アプリUIがリッチメディアを正しく表示できるようになります。

プライベートバケットの場合、パネルがアプリUIで表示されないことがあります。企業にVPNがある場合は、バケットのアクセスポリシーを更新して、VPN内のIPをホワイトリストに登録することができます。
:::
### リファレンスアーティファクトをダウンロードする

```python
import wandb

run = wandb.init()
artifact = run.use_artifact('mnist:latest', type='dataset')
artifact_dir = artifact.download()
```
Weights & Biasesは、アーティファクトが記録された際に記録されたメタデータを使用して、基礎となるバケットから参照されるアーティファクトをダウンロードします。バケットでオブジェクトのバージョン管理が有効になっている場合、Weights & Biasesは、アーティファクトが記録された時点のファイルの状態に対応するオブジェクトのバージョンを取得します。これにより、バケットの内容を進化させるにつれて、アーティファクトがトレーニング時にバケットのスナップショットとして機能するため、特定のモデルがトレーニングされたデータの正確なイテレーションを指すことができます。

:::info
ワークフローの一部としてファイルを上書きする場合、W&BはAmazon S3やGCSバケットで「オブジェクトのバージョン管理」を有効にすることをお勧めします。バージョン管理を有効にすることで、上書きされたファイルへの参照を持つアーティファクトも、古いオブジェクトのバージョンが保持されているため、依然として完全な状態が保たれます。
:::
### まとめ

以下のコード例は、Amazon S3 や GCS に格納されているデータセットをトラッキングし、それをトレーニングジョブに組み込むための簡単なワークフローを示しています：

```python
 import wandb

run = wandb.init()
artifact = wandb.Artifact('mnist', type='dataset')
artifact.add_reference('s3://my-bucket/datasets/mnist')

# アーティファクトをトラッキングし、
# このrunの入力としてマークします。
# バケット内のファイルが変更された場合にのみ、
# 新しいアーティファクトのバージョンがログされます。
run.use_artifact(artifact)
artifact_dir = artifact.download()

# ここでトレーニングを実行...
```

モデルをトラッキングするために、トレーニングスクリプトがモデルファイルをバケットにアップロードした後、モデルのアーティファクトをログに記録することができます：

```python
import boto3
import wandb

run = wandb.init()
# ここでトレーニング...

s3_client = boto3.client('s3')
s3_client.upload_file(
    'my_model.h5',
    'my-bucket',
    'models/cnn/my_model.h5'
    )
モデルアーティファクト = wandb.Artifact（'cnn'、type = 'model'）
モデルアーティファクト.add_reference（'s3://my-bucket/models/cnn/'）
run.log_artifact（モデルアーティファクト）
```

GCPで参照ファイルのトラッキング例、コードとスクリーンショット付きは、[参照によるアーティファクトのトラッキングガイド](https://wandb.ai/stacey/artifacts/reports/Tracking-Artifacts-by-Reference--Vmlldzo1NDMwOTE)を参照してください。
### ファイルシステムの参照

データセットへの迅速なアクセスのための一般的なパターンのもうひとつは、トレーニングジョブを実行しているすべてのマシンでリモートファイルシステムへのNFSマウントポイントを公開することです。これはクラウドストレージバケットよりもさらにシンプルな解決策となることがあります。なぜなら、トレーニングスクリプトの視点から見ると、そのファイルはまるでローカルファイルシステムにあるかのように見えるからです。幸いなことに、ファイルシステムへの参照を追跡するアーティファクトを使って、その使いやすさが拡張されています - マウントされたものであってもそうでなくても。

`/mount` に以下の構造でファイルシステムがマウントされているとしましょう。

```
mount
+-- datasets/
|		+-- mnist/
+-- models/
		+-- cnn/
```
`mnist/`の下には、画像のコレクションであるデータセットがあります。それをアーティファクトでトラッキングしましょう。

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact('mnist', type='dataset')
artifact.add_reference('file:///mount/datasets/mnist/')
run.log_artifact(artifact)
```
デフォルトでは、W&Bはディレクトリへの参照を追加する際に、10,000ファイルの制限を設けています。この制限は、`add_reference`への呼び出しで`max_objects=`を指定することで調整できます。

URLのトリプルスラッシュに注目してください。最初の部分は、ファイルシステムの参照を使用することを示す`file://`プリフィックスです。2つ目は、データセットのパスであり、 `/mount/datasets/mnist/`です。

この結果として得られるアーティファクト`mnist:latest`は、通常のアーティファクトとまったく同じように見え、操作できます。唯一の違いは、アーティファクトはファイルに関するメタデータのみで構成されており、ファイル自体はシステムから離れることはありません。

このアーティファクトは、通常のアーティファクトと同じように操作できます。UIでは、ファイルブラウザを使って参照アーティファクトの内容を閲覧したり、完全な依存関係グラフを調べたり、アーティファクトのバージョン履歴をスキャンしたりできます。ただし、データ自体がアーティファクト内に含まれていないため、UIでは画像やオーディオなどのリッチメディアを表示することはできません。

参照アーティファクトのダウンロードは簡単です：

```python
import wandb

run = wandb.init()
artifact = run.use_artifact(
    'entity/project/mnist:latest', 
    type='dataset'
    )
artifact_dir = artifact.download()
```
ファイルシステムの参照について、「download()」操作は、参照されたパスからファイルをコピーしてアーティファクトのディレクトリーを構築します。上記の例では、`/mount/datasets/mnist`の内容が `artifacts/mnist:v0/`ディレクトリーにコピーされます。アーティファクトに上書きされたファイルへの参照が含まれている場合、`download()`はエラーをスローし、アーティファクトを再構築できなくなります。

すべてをまとめると、以下は、マウントされたファイルシステムの下にあるデータセットをトラッキングし、トレーニングジョブにフィードするためのシンプルなワークフローです。

```python
import wandb
```
runによる初期化：

run = wandb.init()

アーティファクトの作成と参照の追加：

artifact = wandb.Artifact('mnist', type='dataset')
artifact.add_reference('file:///mount/datasets/mnist/')

# アーティファクトをトラッキングし、入力としてマークする

# このrunは一度に実行されます。新しいアーティファクトバージョン
# は、ディレクトリー内のファイルが変更された場合にのみ、
# ログに記録されます。
run.use_artifact(artifact)

artifact_dir = artifact.download()
ここでトレーニングを実行してください...
```

モデルをトラッキングするために、トレーニングスクリプトがモデルファイルをマウントポイントに書き込んだ後に、モデルアーティファクトをログに記録することができます：

```python
import wandb
run = wandb.init()

# ここでトレーニング...

with open('/mount/cnn/my_model.h5') as f:
	# モデルファイルを出力します。
モデルアーティファクト = wandb.Artifact('cnn', type='model')
モデルアーティファクト.add_reference('file:///mount/cnn/my_model.h5')
run.log_artifact(モデルアーティファクト)
```