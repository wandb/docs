---
title: チュートリアル：Docker で W&B ローンンチをセットアップする
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-docker
    parent: set-up-launch
url: guides/launch/setup-launch-docker
---

以下のガイドでは、W&B Launch をローカルマシンで Docker を利用する方法について説明します。ここでは launch エージェントの環境、キューのターゲットリソースの両方に Docker を使用します。

ラウンチエージェントの環境として、またジョブの実行にも同じローカルマシンで Docker を使うのは、Kubernetes のようなクラスター管理システムがインストールされていないマシンでも活用できます。

Docker キューを使えば、パワフルなワークステーション等でワークロードを実行することも可能です。

{{% alert %}}
このセットアップは、自分のローカルマシンで実験するユーザーや、リモートマシンに SSH 接続して launch ジョブを提出するユーザーによく使われます。
{{% /alert %}}

Docker を W&B Launch と組み合わせて利用する場合、まずイメージをビルドし、そのイメージからコンテナを作成・実行します。イメージのビルドには、Docker の `docker run <image-uri>` コマンドが使用されます。キューの設定で定義した内容は、`docker run` コマンドに追加で引数として渡されます。




## Docker キューの設定方法


launch キュー設定（Docker ターゲットリソース用）は、[`docker run`]({{< relref path="/ref/cli/wandb-docker-run.md" lang="ja" >}}) CLI コマンドで定義されているオプションと同じものが利用できます。

エージェントは、キュー設定で定義されたオプションを受け取り、そこへ launch ジョブの設定で上書きされた内容をマージして、最終的な `docker run` コマンドを生成しターゲットリソース（この場合はローカルマシン）で実行します。

この際、2 つの構文変換が発生します：

1. 繰り返し指定できるオプションは、リスト形式でキュー設定に定義します。
2. フラグオプションは、値を `true` にして設定します。

例えば、以下のようなキュー設定の場合：

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

以下のような `docker run` コマンドが生成されます：

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

ボリューム（`volume`）は、文字列リストか単一の文字列として指定できます。複数のボリュームを指定したい場合はリストを使ってください。

Docker では、値が指定されていない環境変数は、launch エージェントの環境から自動的にコンテナへ引き継がれます。つまり、launch エージェントに `MY_EXISTING_ENV_VAR` という環境変数が設定されていれば、その環境変数がコンテナ内でも利用できます。これは、キュー設定で公開したくない config キーを使いたい場合に便利です。

`docker run` コマンドの `--gpus` フラグを利用すれば、Docker コンテナで利用可能な GPU を指定できます。`gpus` フラグの詳細は [Docker のドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu) をご参照ください。


{{% alert %}}
* Docker コンテナ内で GPU を利用する場合は、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) をインストールしてください。
* コードや Artifact をソースにしてイメージをビルドする場合、[エージェント]({{< relref path="#configure-a-launch-agent-on-a-local-machine" lang="ja" >}}) で利用するベースイメージに NVIDIA Container Toolkit が含まれるものを上書き指定できます。たとえば、launch キュー内でベースイメージを `tensorflow/tensorflow:latest-gpu` に指定できます：

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

W&B CLI を使い、Docker を計算リソースとしたキューを作成します：

1. [Launch ページ](https://wandb.ai/launch) にアクセスします。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキュー名を入力します。
5. **Resource** に **Docker** を選択します。
6. **Configuration** フィールドに Docker キュー設定を記述します。
7. **Create Queue** ボタンをクリックしてキューを作成します。

## ローカルマシンで launch エージェントを設定

YAML 設定ファイル `launch-config.yaml` を用意して launch エージェントを設定します。デフォルトでは `~/.config/wandb/launch-config.yaml` が参照されます。エージェント起動時にディレクトリーを指定して、別の設定ファイルを使うことも可能です。

{{% alert %}}
エージェントの主要なオプション（最大同時ジョブ数、Entity、launch キュー）は、設定 YAML ファイルの代わりに W&B CLI から直接指定しても構いません。詳細は [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドをご覧ください。
{{% /alert %}}


## 基本的なエージェント設定オプション

以下のタブでは、W&B CLI と YAML 設定ファイルの両方でエージェントのコア設定オプションを指定する方法を説明します：

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

ローカルマシン上の launch エージェントは、Docker イメージをビルドするように設定できます。デフォルトでは、イメージはマシンのローカルイメージリポジトリに保存されます。launch エージェントで Docker イメージをビルドしたい場合は、launch エージェント設定の `builder` キーを `docker` に設定してください：

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェントが Docker イメージをビルドせず、すでにあるレジストリ上のイメージを使いたい場合は、`builder` キーを `noop` に設定します

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリ

Launch では、Docker Hub、Google Container Registry、Azure Container Registry、Amazon ECR などの外部コンテナレジストリを利用しています。  
もしビルドした環境とは異なる環境でジョブを実行したい場合は、エージェントがコンテナレジストリからイメージを pull できるように設定してください。 

launch エージェントをクラウドレジストリと連携する方法の詳細は、[高度なエージェント設定]({{< relref path="./setup-agent-advanced.md#agent-configuration" lang="ja" >}}) ページをご覧ください。