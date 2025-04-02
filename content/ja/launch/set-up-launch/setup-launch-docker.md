---
title: 'Tutorial: Set up W&B Launch with Docker'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-docker
    parent: set-up-launch
url: guides/launch/setup-launch-docker
---

以下のガイドでは、 W&B Launch を設定して、 ローンチ エージェント環境とキューのターゲットリソースの両方でローカルマシン上の Docker を使用する方法について説明します。

ジョブの実行に Docker を使用すること、および同じローカルマシン上で ローンチ エージェントの環境として使用することは、お使いのコンピューティングが （Kubernetes などの） クラスター 管理システムを持たないマシンにインストールされている場合に特に役立ちます。

また、 Docker キューを使用して、強力なワークステーションでワークロードを実行することもできます。

{{% alert %}}
この設定は、ローカルマシンで実験を実行するユーザーや、SSH で接続して ローンチ ジョブを送信するリモートマシンを持つユーザーによく見られます。
{{% /alert %}}

W&B Launch で Docker を使用すると、W&B は最初にイメージを構築し、次にそのイメージからコンテナを構築して実行します。イメージは、Docker `docker run <image-uri>` コマンドで構築されます。キュー構成は、 `docker run` コマンドに渡される追加の 引数 として解釈されます。

## Docker キューの構成

（Docker ターゲットリソースの） ローンチ キュー構成は、 [`docker run`]({{< relref path="/ref/cli/wandb-docker-run.md" lang="ja" >}}) CLI コマンドで定義されているものと同じオプションを受け入れます。

エージェント は、キュー構成で定義されたオプションを受け取ります。次に、 エージェント は、受信したオプションを ローンチ ジョブの構成からのオーバーライドとマージして、ターゲットリソース （この場合はローカルマシン） で実行される最終的な `docker run` コマンドを生成します。

次の 2 つの構文変換が行われます。

1. 繰り返されるオプションは、キュー構成でリストとして定義されます。
2. フラグオプションは、キュー構成で値が `true` のブール値として定義されます。

たとえば、次のキュー構成があるとします。

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

次の `docker run` コマンドになります。

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

ボリュームは、文字列のリストまたは単一の文字列として指定できます。複数のボリュームを指定する場合は、リストを使用します。

Docker は、値が割り当てられていない 環境 変数を ローンチ エージェント環境から自動的に渡します。つまり、 ローンチ エージェントに 環境 変数 `MY_EXISTING_ENV_VAR` がある場合、その 環境 変数はコンテナで使用できます。これは、キュー構成で公開せずに他の構成 キー を使用する場合に役立ちます。

`docker run` コマンドの `--gpus` フラグを使用すると、Docker コンテナで使用できる GPU を指定できます。 `gpus` フラグの使用方法の詳細については、 [Docker のドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu) を参照してください。

{{% alert %}}
* Docker コンテナ内で GPU を使用するには、 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) をインストールします。
* コードまたは Artifacts ソースのジョブからイメージを構築する場合、 [エージェント ]({{< relref path="#configure-a-launch-agent-on-a-local-machine" lang="ja" >}}) で使用されるベースイメージをオーバーライドして、NVIDIA Container Toolkit を含めることができます。
  たとえば、 ローンチ キュー内で、ベースイメージを `tensorflow/tensorflow:latest-gpu` にオーバーライドできます。

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

## キューの作成

W&B CLI を使用して、Docker をコンピューティングリソースとして使用するキューを作成します。

1. [Launch page](https://wandb.ai/launch)に移動します。
2. [**Create Queue**] ボタンをクリックします。
3. キューを作成する **Entities** を選択します。
4. [**Name**] フィールドにキューの名前を入力します。
5. [**Resource**] として **Docker** を選択します。
6. [**Configuration**] フィールドで Docker キュー構成を定義します。
7. [**Create Queue**] ボタンをクリックしてキューを作成します。

## ローカルマシンでの ローンチ エージェント の構成

`launch-config.yaml` という名前の YAML 構成ファイルを使用して、 ローンチ エージェント を構成します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` で構成ファイルを確認します。オプションで、 ローンチ エージェント をアクティブ化するときに別の ディレクトリー を指定できます。

{{% alert %}}
W&B CLI を使用して、 ローンチ エージェント のコア構成可能オプション （ジョブの最大数、W&B Entity、 ローンチ キュー） を指定できます （構成 YAML ファイルの代わりに）。詳細については、 [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドを参照してください。
{{% /alert %}}

## コア エージェント 構成オプション

次のタブは、W&B CLI および YAML 構成ファイルを使用して、コア構成 エージェント オプションを指定する方法を示しています。

{{< tabpane text=true >}}
{{% tab "W&B CLI" %}}
```bash
wandb launch-agent -q <queue-name> --max-jobs <n>
```
{{% /tab %}}
{{% tab "Config file" %}}
```yaml title="launch-config.yaml"
max_jobs: <n concurrent jobs>
queues:
	- <queue-name>
```
{{% /tab %}}
{{< /tabpane >}}

## Docker イメージビルダー

マシン上の ローンチ エージェント は、Docker イメージを構築するように構成できます。デフォルトでは、これらのイメージはマシンのローカルイメージリポジトリーに保存されます。 ローンチ エージェント が Docker イメージを構築できるようにするには、 ローンチ エージェント 構成の `builder` キー を `docker` に設定します。

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェント に Docker イメージを構築させたくない場合は、代わりにレジストリーから事前に構築されたイメージを使用し、 ローンチ エージェント 構成の `builder` キー を `noop` に設定します。

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリ

Launch は、 Dockerhub、Google Container Registry、Azure Container Registry、Amazon ECR などの外部コンテナレジストリを使用します。
ジョブを構築した環境とは異なる環境でジョブを実行する場合は、コンテナレジストリからプルできるように エージェント を構成します。

ローンチ エージェント を クラウド レジストリに接続する方法の詳細については、 [高度な エージェント のセットアップ]({{< relref path="./setup-agent-advanced.md#agent-configuration" lang="ja" >}}) ページを参照してください。
