---
title: ローンチのセットアップ
menu:
  launch:
    identifier: ja-launch-set-up-launch-_index
    parent: launch
weight: 3
---

このページでは、W&B Launch をセットアップするために必要な高レベルの手順について説明します。

1. **キューのセットアップ**: キューは FIFO 構造で、キュー設定を持ちます。キューの設定は、ターゲットリソース上でジョブがどこでどのように実行されるかを制御します。
2. **エージェントのセットアップ**: エージェントはあなたのマシンやインフラストラクチャー上で動作し、1つ以上のキューをポーリングしてローンチジョブを取得します。ジョブを受け取ると、エージェントはイメージがビルドされ利用可能であることを確認します。そして、ターゲットリソースにジョブを送信します。

## キューのセットアップ
Launch キューは、特定のターゲットリソースを指すように、またそのリソースに固有の追加設定と共に設定する必要があります。例えば、Kubernetes クラスターに向けた Launch キューでは、環境変数やカスタムの namespace を launch キュー設定に含める場合があります。キューを作成する際に、使用したいターゲットリソースと、そのリソース用の設定を指定します。

エージェントがキューからジョブを受け取る際、キュー設定も一緒に受け取ります。エージェントがジョブをターゲットリソースに送信する際、キュー設定とジョブ独自の上書き設定を含めて送信します。例えば、ジョブ設定を使用して、そのジョブだけの Amazon SageMaker インスタンスタイプを指定することも可能です。この場合、[queue config テンプレート]({{< relref path="./setup-queue-advanced.md#configure-queue-template" lang="ja" >}}) をエンドユーザーインターフェースとしてよく使用します。

### キューを作成する
1. [wandb.ai/launch](https://wandb.ai/launch) の Launch App にアクセスします。 
2. 画面右上の **create queue** ボタンをクリックします。

{{< img src="/images/launch/create-queue.gif" alt="Creating a Launch queue" >}}

3. **Entity** ドロップダウンメニューから、キューを所属させたい Entity を選択します。
4. **Queue** フィールドにキューの名前を入力します。
5. **Resource** ドロップダウンから、このキューに追加するジョブで利用したいコンピュートリソースを選択します。
6. このキューで **Prioritization**（優先順位付け）を許可するかどうかを選びます。優先順位を有効にすると、チーム内のユーザーはローンチジョブをキューに追加する際に優先順位を定義でき、高い優先順位のジョブが先に実行されます。
7. **Configuration** フィールドに、JSON か YAML 形式でリソース設定を入力します。設定ドキュメントの構造や意味は、そのキューが指すリソースタイプによって異なります。詳細は、ご利用のターゲットリソースのセットアップ専用ページをご覧ください。

## ローンチエージェントのセットアップ
Launch エージェントは、1つ以上の Launch キューからジョブをポーリングし続けるプロセスです。Launch エージェントは、キューの種類によって FIFO（先入れ先出し）または優先順位順でジョブを取り出します。エージェントはキューからジョブを取り出すと、必要に応じてそのジョブ用のイメージをビルドします。エージェントは、キュー設定で指定されたオプションと共に、ジョブをターゲットリソースに送信します。

{{% alert %}}
エージェントは非常に柔軟で、さまざまなユースケースに対応するよう設定が可能です。エージェントに必要な設定は、あなたの具体的なユースケースによって異なります。[Docker]({{< relref path="./setup-launch-docker.md" lang="ja" >}})、[Amazon SageMaker]({{< relref path="./setup-launch-sagemaker.md" lang="ja" >}})、[Kubernetes]({{< relref path="./setup-launch-kubernetes.md" lang="ja" >}})、[Vertex AI]({{< relref path="./setup-vertex.md" lang="ja" >}}) の各専用ページもご覧ください。
{{% /alert %}}

{{% alert %}}
W&B では、エージェントの起動に特定ユーザーの APIキー ではなく、サービスアカウントの APIキー を使用することを推奨します。サービスアカウントの APIキー を使うメリットは二つあります:
1. エージェントが個々のユーザーに依存しなくなります。
2. Launch で作成された run の author は、エージェントに紐づくユーザーではなく、そのローンチジョブを送信したユーザーとして Launch から認識されます。
{{% /alert %}}

### エージェントの設定
YAML ファイル `launch-config.yaml` を使用してローンチエージェントの設定を行います。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` 内の設定ファイルを参照します。エージェントを起動する際に、別のディレクトリーを指定することも可能です。

エージェントの設定ファイルの内容は、エージェントが動作する環境、Launch キューのターゲットリソース、Dockerビルダーの要件、クラウドレジストリの要件などによって異なります。

ユースケースにかかわらず、Launch エージェントには共通して設定できる主要なオプションがあります:
* `max_jobs`: エージェントが並列に処理できる最大ジョブ数 
* `entity`: キューの所属する Entity
* `queues`: エージェントが監視する1つ以上のキュー名リスト

{{% alert %}}
W&B CLI を使って、設定 YAML ファイルではなく、Launch エージェントの主要な設定オプション（最大ジョブ数、W&B Entity、Launch キュー）を指定できます。詳細は [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドのドキュメントをご覧ください。
{{% /alert %}}


以下は、主要な Launch エージェント設定キーを指定する YAML スニペット例です:

```yaml title="launch-config.yaml"
# 同時実行できる最大 run 数。-1 = 制限なし
max_jobs: -1

entity: <entity-name>

# ポーリングするキューのリスト
queues:
  - <queue-name>
```

### コンテナービルダーの設定
ローンチエージェントはイメージをビルドするように設定できます。git リポジトリやコードアーティファクトから作成したローンチジョブを利用する場合、エージェントでコンテナービルダーを設定する必要があります。ローンチジョブの作成方法については [Create a launch job]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) をご覧ください。

W&B Launch では、次の3つのビルダーオプションをサポートしています:

* Docker: ローカルの Docker デーモンを使ってイメージをビルドします。
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Docker デーモンがない環境でもイメージビルドを可能にする Google のプロジェクトです。
* Noop: エージェントはビルドを行わず、事前にビルド済みのイメージのみを取得します。

{{% alert %}}
Docker デーモンが使えない環境（例: Kubernetes クラスター）でエージェントをポーリングする場合は、Kaniko ビルダーを使いましょう。

Kaniko ビルダーの詳細は [Set up Kubernetes]({{< relref path="./setup-launch-kubernetes.md" lang="ja" >}}) をご覧ください。
{{% /alert %}}

イメージビルダーを指定するには、エージェントの設定ファイル内に builder キーを追加します。下記のコードスニペットは、Docker または Kaniko を使用する設定例です（`launch-config.yaml` 内の一部）:

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### コンテナーレジストリの設定
場合によっては、ローンチエージェントをクラウドレジストリと連携させたいこともあるでしょう。クラウドレジストリを利用したい典型的なケースは以下の通りです:

* ビルドしたジョブを、そのジョブをビルドした環境とは異なる環境（高性能ワークステーションやクラスターなど）で実行したい
* エージェントでイメージをビルドし、Amazon SageMaker や Vertex AI 上でこれらを実行したい
* イメージリポジトリから取得するための認証情報をローンチエージェントで提供したい

コンテナーレジストリとの連携方法について詳しくは、[Advanced agent set]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) up ページをご覧ください。

## ローンチエージェントの起動
W&B CLI の `launch-agent` コマンドを使ってローンチエージェントを起動します:

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

場合によっては、Kubernetes クラスター内からローンチエージェントにキューをポーリングさせたいこともあるでしょう。さらに詳しくは [Advanced queue set up page]({{< relref path="./setup-queue-advanced.md" lang="ja" >}}) をご覧ください。