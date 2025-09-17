---
title: 'チュートリアル: Docker で W&B Launch をセットアップする'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-docker
    parent: set-up-launch
url: guides/launch/setup-launch-docker
---

以下のガイドでは、同一のローカルマシン上で Docker を、Launch エージェントの環境およびキューのターゲットリソースの両方に使うように W&B Launch を設定する方法を説明します。

クラスター 管理システム（Kubernetes など）がないマシンに計算環境が入っている場合、ジョブの実行と Launch エージェントの環境の両方に Docker を同一マシンで用いる方法は特に便利です。

また、Docker キューを使ってパワフルなワークステーションでワークロードを実行することもできます。

{{% alert %}}
このセットアップは、ローカルマシンで実験を行う ユーザー、または SSH でログインして Launch ジョブを送信するリモートマシンを持つ ユーザー によく見られます。
{{% /alert %}}

W&B Launch で Docker を使うと、まずイメージをビルドし、そのイメージからコンテナをビルドして実行します。イメージは Docker の `docker run <image-uri>` コマンドでビルドされます。キューの設定は、`docker run` コマンドに渡される追加の引数として解釈されます。




## Docker キューを設定する

Launch キューの設定（Docker ターゲットリソース用）は、[`docker run`]({{< relref path="/ref/cli/wandb-docker-run.md" lang="ja" >}}) CLI コマンドで定義されているものと同じオプションを受け付けます。

エージェントは、キューの設定で定義されたオプションを受け取ります。次にエージェントは、それらと Launch ジョブの設定による上書き指定をマージして、最終的にターゲットリソース（この場合はローカルマシン）で実行される `docker run` コマンドを構築します。

次の 2 つの構文変換が行われます:
1. 繰り返し指定できるオプションは、キュー設定ではリストで定義します。
2. フラグオプションは、キュー設定では `true` のブール値として定義します。

例えば、次のようなキュー設定では:

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

次のような `docker run` コマンドになります:

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

ボリュームは、文字列のリストまたは単一の文字列として指定できます。複数のボリュームを指定する場合はリストを使用してください。

Docker は、値を割り当てていない環境変数を Launch エージェントの環境から自動的に引き継ぎます。つまり、Launch エージェントに `MY_EXISTING_ENV_VAR` という環境変数があれば、その環境変数はコンテナ内でも利用できます。これは、キューの設定に公開せずに他の設定 キー を使いたい場合に便利です。

`docker run` コマンドの `--gpus` フラグを使うと、Docker コンテナで使用可能な GPU を指定できます。`gpus` フラグの使い方の詳細は [Docker のドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu) を参照してください。


{{% alert %}}
* Docker コンテナ内で GPU を使うには、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) をインストールしてください。
* コードまたは Artifacts 由来のジョブからイメージをビルドする場合、[エージェント]({{< relref path="#configure-a-launch-agent-on-a-local-machine" lang="ja" >}}) が使うベースイメージを上書きして NVIDIA Container Toolkit を含めることができます。
  例えば、Launch キュー内でベースイメージを `tensorflow/tensorflow:latest-gpu` に上書きできます:

  ```json
  {
    "builder": {
      "accelerator": {
        "base_image": "tensorflow/tensorflow:latest-gpu"
      }
    }
  }
  ```
{{% /alert %}}




## キューを作成する

W&B CLI を使って、計算リソースに Docker を用いるキューを作成します:

1. [Launch page](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成する **Entity** を選択します。
4. **Name** フィールドにキュー名を入力します。
5. **Resource** として **Docker** を選択します。
6. **Configuration** フィールドに Docker キューの設定を記入します。
7. **Create Queue** をクリックしてキューを作成します。

## ローカルマシンで Launch エージェントを設定する

Launch エージェントは `launch-config.yaml` という名前の YAML 設定ファイルで設定します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` に設定ファイルがあるかを確認します。Launch エージェントを起動するときに、別の ディレクトリー を指定することもできます。

{{% alert %}}
W&B CLI を使って、Launch エージェントの主要な設定可能オプション（YAML ファイルの代わりに）を指定できます。指定できるのは、ジョブの最大数、W&B の Entity、Launch キューです。詳細は [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドを参照してください。
{{% /alert %}}


## エージェントの基本設定オプション

以下のタブは、W&B CLI と YAML 設定ファイルで基本のエージェント設定オプションを指定する方法を示します:

{{< tabpane text=true >}}
{{% tab "W&B CLI" %}}
```bash
wandb launch-agent -q <queue-name> --max-jobs <n>
```
{{% /tab %}}
{{% tab "設定ファイル" %}}
```yaml title="launch-config.yaml"
max_jobs: <n concurrent jobs>
queues:
	- <queue-name>
```
{{% /tab %}}
{{< /tabpane >}}

## Docker イメージビルダー

お使いのマシン上の Launch エージェントは、Docker イメージをビルドするように設定できます。デフォルトでは、これらのイメージはマシンのローカルイメージリポジトリーに保存されます。Launch エージェントの設定で `builder` キー を `docker` に設定すると、Docker イメージのビルドが有効になります:

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェントに Docker イメージをビルドさせず、レジストリーの事前ビルドイメージを使用したい場合は、Launch エージェントの設定で `builder` キー を `noop` に設定してください

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリー

Launch は Dockerhub、Google Container Registry、Azure Container Registry、Amazon ECR などの外部コンテナレジストリーを使用します。  
ビルドした 環境 とは異なる 環境 でジョブを実行したい場合は、コンテナレジストリーから pull できるようにエージェントを設定してください。 

Launch エージェントを クラウド レジストリーに接続する方法の詳細は、[高度なエージェント設定]({{< relref path="./setup-agent-advanced.md#agent-configuration" lang="ja" >}}) のページを参照してください。