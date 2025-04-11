---
title: 'チュートリアル: Docker を使用して W&B ローンンチを設定する'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-docker
    parent: set-up-launch
url: guides/launch/setup-launch-docker
---

以下のガイドでは、ローカルマシンでドッカーを使用してW&B Launchを設定し、ローンンチエージェントの環境とキューの対象リソースとして使用する方法を説明します。

ドッカーを使用してジョブを実行し、同じローカルマシン上でローンンチエージェントの環境として使用することは、クラスター管理システム（Kubernetesなど）がインストールされていないマシンにコンピュートがある場合に特に役立ちます。

また、強力なワークステーションでのワークロードを実行するためにドッキューを使用することもできます。

{{% alert %}}
このセットアップは、実験をローカルマシンで行うユーザーや、SSHでアクセスしてローンンチジョブを提出するリモートマシンを持っているユーザーにとって一般的です。
{{% /alert %}}

ドッカーをW&B Launchと一緒に使用する場合、W&Bはまずイメージをビルドし、そのイメージからコンテナをビルドして実行します。イメージはDockerの `docker run <image-uri>` コマンドでビルドされます。キュー設定は、`docker run` コマンドに渡される追加の引数として解釈されます。

## Dockerキューの設定

Dockerの対象リソースのためのローンンチキュー設定は、[`docker run`]({{< relref path="/ref/cli/wandb-docker-run.md" lang="ja" >}}) CLIコマンドで定義された同じオプションを受け入れます。

エージェントはキュー設定で定義されたオプションを受け取り、その後、受け取ったオプションをローンンチジョブの設定からのオーバーライドとマージし、対象リソース（この場合はローカルマシン）で実行される最終的な `docker run` コマンドを生成します。

次の2つの構文変換が行われます：

1. 繰り返しオプションは、キュー設定でリストとして定義されます。
2. フラグオプションは、キュー設定で `true` の値を持つブール値として定義されます。

例えば、以下のキュー設定：

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

結果として以下の `docker run` コマンドになります：

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

ボリュームは文字列のリストまたは単一の文字列として指定できます。複数のボリュームを指定する場合はリストを使用してください。

ドッカーは値が割り当てられていない環境変数をローンンチエージェントの環境から自動的に渡します。これは、ローンンチエージェントが環境変数 `MY_EXISTING_ENV_VAR` を持っている場合、その環境変数がコンテナ内で利用可能であることを意味します。これは、キュー設定で公開せずに他の設定キーを使用したい場合に便利です。

`docker run` コマンドの `--gpus` フラグを使って、ドッカーコンテナで利用可能なGPUを指定できます。`gpus` フラグの使用方法について詳しくは、[Dockerのドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu)をご覧ください。

{{% alert %}}
* Dockerコンテナ内でGPUを使用するには、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) をインストールしてください。
* コードやアーティファクトをソースとするジョブからイメージをビルドする場合、NVIDIA Container Toolkitを含めるために[エージェント]({{< relref path="#configure-a-launch-agent-on-a-local-machine" lang="ja" >}})でベースイメージをオーバーライドできます。
  例えば、キュー内でベースイメージを `tensorflow/tensorflow:latest-gpu` にオーバーライドできます：

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

W&B CLIを使用して、ドッカーをコンピュートリソースとして使用するキューを作成します：

1. [Launchページ](https://wandb.ai/launch)に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **Docker** を選択します。
6. **Configuration** フィールドにドッカーキューの設定を定義します。
7. **Create Queue** ボタンをクリックしてキューを作成します。

## ローカルマシンでローンンチエージェントを設定する

ローンンチエージェントを `launch-config.yaml` という名前の YAML 設定ファイルで設定します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` で設定ファイルをチェックします。ローンンチエージェントをアクティベートするときに異なるディレクトリをオプションとして指定できます。

{{% alert %}}
ローンンチエージェントのために、W&B CLIを使用して、最大ジョブ数、W&Bエンティティ、およびローンンチキューの主要な設定可能なオプションを指定できます（config YAMLファイルではなく）。[`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドを参照してください。
{{% /alert %}}

## コアエージェント設定オプション

以下のタブは、W&B CLIとYAML設定ファイルを使用して、コアエージェント設定オプションを指定する方法を示しています：

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

## ドッカーイメージビルダー

マシン上のローンンチエージェントは、ドッカーイメージをビルドするように設定できます。デフォルトでは、これらのイメージはマシンのローカルイメージリポジトリに保存されます。ローンンチエージェントがドッカーイメージをビルドできるようにするには、ローンンチエージェント設定で `builder` キーを `docker` に設定します：

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェントがドッカーイメージをビルドせず、代わりにレジストリからの事前ビルドイメージを使用したい場合は、ローンンチエージェント設定で `builder` キーを `noop` に設定します：

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリ

Launch は Dockerhub、Google Container Registry、Azure Container Registry、Amazon ECR などの外部コンテナレジストリを使用します。
異なる環境でジョブを実行したい場合は、コンテナレジストリから引き出せるようにエージェントを設定してください。

ローンンチエージェントをクラウドレジストリと接続する方法について詳しくは、[Advanced agent setup]({{< relref path="./setup-agent-advanced.md#agent-configuration" lang="ja" >}}) ページを参照してください。