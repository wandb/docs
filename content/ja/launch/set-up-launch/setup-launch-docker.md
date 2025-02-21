---
title: 'Tutorial: Set up W&B Launch with Docker'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-docker
    parent: set-up-launch
url: guides/launch/setup-launch-docker
---

W&B Launch の設定方法について、Docker をローカルマシンで使用して、launch agent environment と queue のターゲットリソースを設定するガイドです。

Docker を使用してジョブを実行し、launch agent の環境を同じローカルマシンで設定すると、クラスター管理システム（例えば Kubernetes）がインストールされていないマシンにコンピューティングリソースがある場合に特に便利です。

Docker キューを使用して、パワフルなワークステーションでワークロードを実行することもできます。

{{% alert %}}
このセットアップは、ローカルマシンで実験を行うユーザーや、SSH でリモートマシンに接続し、launch ジョブを送信するユーザーによく見られます。
{{% /alert %}}

W&B Launch に Docker を使用する場合、W&B はまずイメージをビルドし、そのイメージからコンテナをビルドおよび実行します。イメージは Docker の `docker run <image-uri>` コマンドでビルドされます。queue の設定は、`docker run` コマンドに渡される追加の引数として解釈されます。

## Docker queue の設定

launch queue の設定（Docker ターゲットリソースの場合）は、[`docker run`]({{< relref path="/ref/cli/wandb-docker-run.md" lang="ja" >}}) CLI コマンドで定義されている同じオプションを受け入れます。

エージェントは queue の設定で定義されたオプションを受け取ります。その後、エージェントは launch ジョブの設定による何らかのオーバーライドと受け取ったオプションをマージし、ターゲットリソース上で実行される最終的な `docker run` コマンドを生成します（この場合、ローカルマシン）。

以下の 2 つの構文変換が行われます：

1. 繰り返しのオプションは、queue の設定でリストとして定義されます。
2. フラグオプションは、queue の設定で Boolean で `true` の値を持つものとして定義されます。

例えば、次の queue 設定は：

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

次の `docker run` コマンドを生成します：

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

ボリュームは、文字列のリストまたは単一の文字列として指定できます。複数のボリュームを指定する場合はリストを使用してください。

値が割り当てられていない環境変数は、launch agent の環境から自動的に Docker によって渡されます。つまり、launch agent が環境変数 `MY_EXISTING_ENV_VAR` を持っている場合、その環境変数はコンテナ内で利用可能です。他の設定キーを queue 設定で公開せずに使用したい場合に便利です。

`docker run` コマンドの `--gpus` フラグを使用すると、Docker コンテナに対して使用可能な GPU を指定できます。`gpus` フラグの使用方法の詳細については、[Docker ドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu)を参照してください。

{{% alert %}}
* Docker コンテナ内で GPU を使用するには、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) をインストールしてください。
* コードまたはアーティファクトソースのジョブからイメージをビルドする場合、[エージェント]({{< relref path="#configure-a-launch-agent-on-a-local-machine" lang="ja" >}})が使用するベースイメージを NVIDIA Container Toolkit を含むものにオーバーライドできます。
  例えば、launch queue 内でベースイメージを `tensorflow/tensorflow:latest-gpu` にオーバーライドすることができます：

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

W&B CLI を使って Docker をコンピューティングリソースとして使用するキューを作成します。

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **Docker** を選択します。
6. **Configuration** フィールドに Docker キュー設定を定義します。
7. **Create Queue** ボタンをクリックしてキューを作成します。

## ローカルマシンでの launch agent の設定

YAML コンフィグファイル `launch-config.yaml` を使って launch agent を設定します。デフォルトでは W&B は `~/.config/wandb/launch-config.yaml` にある設定ファイルをチェックします。launch agent を起動するときに異なるディレクトリーを指定することも可能です。

{{% alert %}}
launch agent のコア設定オプションを W&B CLI で指定できます（設定 YAML ファイルの代わりに）。最大ジョブ数、W&B entity、launch queue があります。詳しくは [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドを参照してください。
{{% /alert %}}

## コアエージェント設定オプション

以下のタブでは、W&B CLI と YAML 設定ファイルを用いてコア設定エージェントオプションを指定する方法を示します：

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

マシン上の launch agent は、Docker イメージをビルドするように設定できます。デフォルトでは、これらのイメージはマシンのローカルイメージリポジトリに保存されます。launch agent が Docker イメージをビルドするようにするには、launch agent コンフィグの `builder` キーを `docker` に設定します：

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェントが Docker イメージをビルドせず、代わりにレジストリから事前ビルドのイメージを使用したい場合、launch agent 設定の `builder` キーを `noop` に設定します：

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリ

Launch は Dockerhub、Google Container Registry、Azure Container Registry、Amazon ECR などの外部コンテナレジストリを使用します。  
構築した環境とは異なる環境でジョブを実行したい場合、エージェントを設定してコンテナレジストリからプルできるようにします。

launch agent をクラウドレジストリと接続する方法の詳細については、[Advanced agent setup]({{< relref path="./setup-agent-advanced.md#agent-configuration" lang="ja" >}}) ページを参照してください。