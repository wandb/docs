---
title: 'チュートリアル: Docker で W&B Launch をセットアップする'
menu:
  launch:
    identifier: setup-launch-docker
    parent: set-up-launch
url: guides/launch/setup-launch-docker
---

以下のガイドは、W&B Launch でローカルマシン上で Docker を launch agent の環境とキューのターゲットリソースの両方に使用する方法を説明します。

同じローカルマシンでジョブの実行と launch agent の環境の両方に Docker を使うのは、Kubernetes などのクラスター管理システムがインストールされていないマシンで計算処理を行う場合に特に便利です。

また、Docker キューを使って、高性能なワークステーション上でワークロードを実行することもできます。

{{% alert %}}
このセットアップは、ローカルマシンで実験を行うユーザーや、リモートマシンに SSH 接続して launch ジョブを送信するユーザーによく使われます。
{{% /alert %}}

W&B Launch で Docker を使う場合、W&B はまずイメージをビルドし、そのイメージからコンテナをビルドして実行します。イメージは Docker の `docker run <image-uri>` コマンドで実行されます。キューの設定は、`docker run` コマンドに追加される引数として解釈されます。

## Docker キューを設定する

Launch キューの設定（Docker をターゲットリソースとする場合）は、[`docker run`]({{< relref "/ref/cli/wandb-docker-run.md" >}}) CLI コマンドで定義されるオプションと同じものを受け付けます。

エージェントは、キュー設定で定義されたオプションを受け取ります。エージェントはその後、受け取ったオプションと launch ジョブの設定からの上書き内容をマージして、最終的な `docker run` コマンドを生成し、ターゲットリソース（ここではローカルマシン）で実行します。

2つのシンタックス変換が行われます：

1. 繰り返し指定するオプションは、キュー設定ではリストとして記載します。
2. フラグオプションは、キュー設定で値を `true` とした Boolean で記載します。

例えば、次のようなキュー設定を考えてみましょう：

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

これにより、以下のような `docker run` コマンドが生成されます：

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

ボリュームは、文字列のリストか、単一の文字列で指定できます。複数のボリュームを指定する場合はリストを使いましょう。

Docker は、値を割り当てていない環境変数も自動的に launch agent の環境から引き継ぎます。つまり、launch agent に `MY_EXISTING_ENV_VAR` という環境変数があれば、その環境変数はコンテナ内でも使えます。これにより、他の config キーをキュー設定で公開せずに利用したい場合に便利です。

`docker run` コマンドの `--gpus` フラグを使うと、Docker コンテナで利用可能な GPU を指定できます。`gpus` フラグの詳細な使い方については [Docker のドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu) を参照してください。

{{% alert %}}
* Docker コンテナ内で GPU を利用するには、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) をインストールしてください。
* コードや Artifact をソースとするジョブからイメージをビルドする際、[agent]({{< relref "#configure-a-launch-agent-on-a-local-machine" >}}) で使用するベースイメージを NVIDIA Container Toolkit を含むものに上書きできます。
  例えば、launch キュー設定内でベースイメージを `tensorflow/tensorflow:latest-gpu` に変更することができます：

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

W&B CLI で計算リソースとして Docker を使うキューを作成する手順です：

1. [Launch ページ](https://wandb.ai/launch) にアクセスします。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** 欄にキューの名前を入力します。
5. **Resource** として **Docker** を選択します。
6. **Configuration** 欄に Docker キューの設定を記入します。
7. **Create Queue** ボタンをクリックしてキューを作成します。

## ローカルマシンで launch agent を設定する

`launch-config.yaml` という名前の YAML 設定ファイルで launch agent を設定します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` の設定ファイルをチェックします。エージェントを起動するときに、別のディレクトリーを指定することもできます。

{{% alert %}}
W&B CLI を使って launch agent のコア設定オプション（YAML ファイルの代わりに）：最大ジョブ数、W&B Entity、launch キューを指定することもできます。詳しくは [`wandb launch-agent`]({{< relref "/ref/cli/wandb-launch-agent.md" >}}) コマンドを参照してください。
{{% /alert %}}

## コアエージェント設定オプション

以下のタブで、W&B CLI と YAML 設定ファイルの両方でコアエージェント設定オプションを指定する方法を紹介します：

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

あなたのマシン上の launch agent は、Docker イメージをビルドできるように設定できます。デフォルトで、これらのイメージはマシンのローカルイメージリポジトリに保存されます。エージェントで Docker イメージビルドを有効化するには、launch agent 設定で `builder` キーを `docker` に設定します：

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェントに Docker イメージのビルドをさせず、レジストリからの事前ビルド済みイメージを利用したい場合は、launch agent 設定の `builder` キーを `noop` に設定します

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリ

Launch では Dockerhub、Google Container Registry、Azure Container Registry、Amazon ECR など、外部のコンテナレジストリを利用します。  
ビルドした環境と異なる環境でジョブを実行したい場合は、エージェントがコンテナレジストリからイメージを取得できるように設定してください。

launch agent とクラウドレジストリの連携方法について詳しくは、[Advanced agent setup]({{< relref "./setup-agent-advanced.md#agent-configuration" >}}) のページをご覧ください。