---
title: 'Tutorial: Set up W&B Launch with Docker'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-docker
    parent: set-up-launch
url: guides/launch/setup-launch-docker
---

以下のガイドでは、W&B Launch を設定して、ローカルマシン上の Docker を Launch エージェント環境とキューのターゲットリソースの両方に使用する方法について説明します。

ジョブの実行に Docker を使用し、同じローカルマシン上で Launch エージェントの環境としても使用することは、コンピューティングが Kubernetes などのクラスター管理システムを持たないマシンにインストールされている場合に特に役立ちます。

Docker キューを使用して、強力なワークステーションでワークロードを実行することもできます。

{{% alert %}}
この設定は、ローカルマシンで実験を行ったり、SSH で接続するリモートマシンから Launch ジョブを送信したりするユーザーによく見られます。
{{% /alert %}}

W&B Launch で Docker を使用すると、W&B は最初にイメージを構築し、次にそのイメージからコンテナを構築して実行します。イメージは、Docker の `docker run <image-uri>` コマンドで構築されます。キューの設定は、`docker run` コマンドに渡される追加の 引数 として解釈されます。

## Docker キューの設定

Launch キューの設定 (Docker ターゲットリソースの場合) は、[`docker run`]({{< relref path="/ref/cli/wandb-docker-run.md" lang="ja" >}}) CLI コマンドで定義されているものと同じオプションを受け入れます。

エージェント は、キューの設定で定義されたオプションを受け取ります。次に、エージェント は、受信したオプションを Launch ジョブの設定からのオーバーライドとマージして、ターゲットリソース (この場合はローカルマシン) で実行される最終的な `docker run` コマンドを生成します。

次の 2 つの構文変換が行われます。

1. 繰り返されるオプションは、キューの設定でリストとして定義されます。
2. フラグオプションは、キューの設定で Boolean 型で値 `true` として定義されます。

たとえば、次のキュー設定があるとします。

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

Docker は、値が割り当てられていない環境変数を Launch エージェント 環境から自動的に渡します。つまり、Launch エージェント に環境変数 `MY_EXISTING_ENV_VAR` がある場合、その環境変数はコンテナで使用できます。これは、キューの設定で公開せずに他の config キーを使用する場合に役立ちます。

`docker run` コマンドの `--gpus` フラグを使用すると、Docker コンテナで使用できる GPU を指定できます。`gpus` フラグの使用方法の詳細については、[Docker ドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu) を参照してください。

{{% alert %}}
* Docker コンテナ内で GPU を使用するには、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) をインストールします。
* コードまたは Artifacts をソースとするジョブからイメージを構築する場合は、[エージェント ]({{< relref path="#configure-a-launch-agent-on-a-local-machine" lang="ja" >}})で使用されるベースイメージをオーバーライドして、NVIDIA Container Toolkit を含めることができます。
  たとえば、Launch キュー内で、ベースイメージを `tensorflow/tensorflow:latest-gpu` にオーバーライドできます。

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

1. [Launch ページ](https://wandb.ai/launch)に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成する **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **Docker** を選択します。
6. **Configuration** フィールドで Docker キューの設定を定義します。
7. **Create Queue** ボタンをクリックして、キューを作成します。

## ローカルマシンでの Launch エージェント の設定

`launch-config.yaml` という名前の YAML 設定ファイルを使用して、Launch エージェント を設定します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` で設定ファイルを確認します。Launch エージェント をアクティブ化するときに、別のディレクトリーをオプションで指定できます。

{{% alert %}}
W&B CLI を使用して、Launch エージェント の主要な設定可能オプション (ジョブの最大数、W&B Entity 、Launch キュー) を指定できます (設定 YAML ファイルの代わりに)。詳細については、[`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドを参照してください。
{{% /alert %}}

## 主要なエージェント config オプション

次のタブは、W&B CLI と YAML 設定ファイルを使用して、主要な config エージェント オプションを指定する方法を示しています。

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

マシン上の Launch エージェント は、Docker イメージを構築するように設定できます。デフォルトでは、これらのイメージはマシンのローカルイメージリポジトリに保存されます。Launch エージェント が Docker イメージを構築できるようにするには、Launch エージェント config の `builder` キーを `docker` に設定します。

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェント に Docker イメージを構築させたくない場合は、代わりにレジストリから事前構築済みのイメージを使用する場合は、Launch エージェント config の `builder` キーを `noop` に設定します。

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリ

Launch は、Dockerhub、Google Container Registry、Azure Container Registry、Amazon ECR などの外部コンテナレジストリを使用します。
ジョブを構築した環境とは異なる環境でジョブを実行する場合は、コンテナレジストリからプルできるように エージェント を設定します。

Launch エージェント を クラウド レジストリに接続する方法の詳細については、[エージェント の高度な設定]({{< relref path="./setup-agent-advanced.md#agent-configuration" lang="ja" >}}) ページを参照してください。
