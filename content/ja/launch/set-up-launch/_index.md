---
title: Launch をセットアップする
menu:
  launch:
    identifier: setup-launch
    parent: launch
weight: 3
---

このページでは、W&B Launch のセットアップに必要なハイレベルな手順について説明します。

1. **キューのセットアップ**: キューは FIFO （先入れ先出し）であり、キュー設定を持ちます。キューの設定は、どのリソース上でどのようにジョブが実行されるかを制御します。
2. **エージェントのセットアップ**: エージェントはあなたのマシンやインフラストラクチャー上で動作し、1つ以上のキューから launch ジョブをポーリングします。ジョブが取得されると、エージェントはイメージがビルド済みで利用可能であることを確認し、そのジョブをターゲットリソースに提出します。

## キューのセットアップ
Launch キューは、特定のターゲットリソースを指定するように設定し、そのリソース固有の追加設定も行う必要があります。例えば、Kubernetes クラスターを指す launch キューでは、環境変数やカスタムの namespace を launch キュー設定で指定することができます。キューを作成する際には、利用したいターゲットリソースと、そのリソースに対する設定の両方を指定します。

エージェントがキューからジョブを受信するとき、同時にキュー設定も受け取ります。エージェントがジョブをターゲットリソースに提出する際、そのジョブ固有のオーバーライドとともにキュー設定を含めます。例えば、ジョブ設定を使って、特定のジョブインスタンス用に Amazon SageMaker インスタンスタイプを指定することができます。この場合、多くの場合で [queue config templates]({{< relref "./setup-queue-advanced.md#configure-queue-template" >}}) をエンドユーザーインターフェースとして利用します。

### キューの作成
1. [wandb.ai/launch](https://wandb.ai/launch) の Launch App にアクセスします。
2. 画面右上の **create queue** ボタンをクリックします。

{{< img src="/images/launch/create-queue.gif" alt="Creating a Launch queue" >}}

3. **Entity** ドロップダウンメニューから、キューが所属する Entity を選択します。
4. **Queue** フィールドに、作成するキューの名前を入力します。
5. **Resource** ドロップダウンから、このキューに追加するジョブが利用する計算リソースを選択します。
6. このキューで **Prioritization**（優先度指定）を許可するかどうかを選びます。優先度が有効な場合、チームのユーザーが launch ジョブをキューに登録する際、優先度を指定できます。優先度が高いジョブほど先に実行されます。
7. **Configuration** フィールドでリソース設定を JSON または YAML フォーマットで入力します。設定ドキュメントの構造や意味は、キューがポイントしているリソースタイプによって異なります。詳細は、対象リソース毎のセットアップページをご覧ください。

## launch エージェントのセットアップ
launch エージェントは、launch キューからジョブをポーリングし続ける常駐プロセスです。launch エージェントは、取得するキューによって FIFO または優先度順でジョブをデキューします。エージェントがキューからジョブをデキューすると、必要に応じてそのジョブ用のイメージをビルドします。その後、キュー設定で指定されたオプションとともにジョブをターゲットリソースに提出します。

{{% alert %}}
エージェントは非常に柔軟で、多彩なユースケースに対応出来るよう設定できます。必要な設定はユースケースによって異なります。[Docker]({{< relref "./setup-launch-docker.md" >}})、[Amazon SageMaker]({{< relref "./setup-launch-sagemaker.md" >}})、[Kubernetes]({{< relref "./setup-launch-kubernetes.md" >}})、[Vertex AI]({{< relref "./setup-vertex.md" >}}) のセットアップ専用ページもご覧ください。
{{% /alert %}}

{{% alert %}}
W&B では、エージェントを特定のユーザーの API キーではなく、サービスアカウントの API キーで開始することを推奨します。サービスアカウントの API キーを使うと、次の2つのメリットがあります：
1. エージェントが個々のユーザーに依存しなくなります。
2. Launch で作成された run のオーサーが、エージェントのユーザーではなく launch ジョブを提出したユーザーとして Launch で認識されます。
{{% /alert %}}

### エージェントの設定
launch エージェントは `launch-config.yaml` という名前の YAML ファイルで設定します。デフォルトで、W&B は `~/.config/wandb/launch-config.yaml` 内の設定ファイルを参照します。必要に応じて、エージェントを起動する際に別ディレクトリーを指定することもできます。

launch エージェントの設定ファイルの内容は、エージェントの実行環境、launch キューのターゲットリソース、Docker ビルダー要件、クラウドレジストリ要件などによって異なります。

ユースケースを問わず、launch エージェントでコアとなる設定オプションは以下のとおりです：
* `max_jobs`: エージェントが同時に実行できる最大ジョブ数
* `entity`: キューが所属する entity
* `queues`: エージェントが監視する1つ以上のキュー名

{{% alert %}}
W&B CLI でも、launch エージェントの基本的な設定オプション（最大ジョブ数、W&B entity、launch キュー）を指定できます（YAML ファイルの代わりに）。詳細は [`wandb launch-agent`]({{< relref "/ref/cli/wandb-launch-agent.md" >}}) コマンドをご覧ください。
{{% /alert %}}

以下の YAML スニペットは、コアとなる launch エージェントの設定キーの指定例です：

```yaml title="launch-config.yaml"
# 並行して実行する最大 run 数。-1 = 無制限
max_jobs: -1

entity: <entity-name>

# ポーリングするキューのリスト
queues:
  - <queue-name>
```

### コンテナビルダーの設定
launch エージェントはイメージのビルドも可能です。Git リポジトリやコード Artifacts から作成された launch ジョブに対応する場合、エージェントでコンテナビルダーを設定する必要があります。launch ジョブの作成方法については、[Create a launch job]({{< relref "../create-and-deploy-jobs/create-launch-job.md" >}}) をご覧ください。

W&B Launch では3つのビルダーオプションをサポートしています：

* Docker: ローカル Docker デーモンを使ってイメージをビルドします。
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Kaniko は、Docker デーモンが利用できない環境でイメージをビルドできる Google のプロジェクトです。
* Noop: エージェントはジョブのビルドを試みず、事前にビルドされたイメージのみを pull します。

{{% alert %}}
エージェントの実行環境に Docker デーモンが無い場合（例：Kubernetes クラスターなど）は Kaniko ビルダーを利用してください。

Kaniko ビルダーの詳細は、[Set up Kubernetes]({{< relref "./setup-launch-kubernetes.md" >}}) をご覧ください。
{{% /alert %}}

イメージビルダーを指定するには、エージェント設定に builder キーを含めます。たとえば次のコードスニペットは、Docker または Kaniko を指定する launch config（一部）の例です：

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### コンテナレジストリの設定
場合によっては、launch エージェントをクラウドレジストリに接続したいことがあるかもしれません。一般的なユースケースとしては：

* イメージをビルドした場所以外（高性能ワークステーションやクラスターなど）でジョブを実行したい場合
* Amazon SageMaker や VertexAI でイメージをビルドし実行したい場合
* launch エージェントでイメージリポジトリから pull するための認証情報を提供したい場合

エージェントがコンテナレジストリと連携するための設定方法は、[Advanced agent set up page]({{< relref "./setup-agent-advanced.md" >}}) をご覧ください。

## launch エージェントの起動
launch エージェントは `launch-agent` W&B CLI コマンドで起動します：

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

ユースケースによっては、launch エージェントを Kubernetes クラスター内部でキューをポーリングする形で運用したい場合もあります。詳細は [Advanced queue set up page]({{< relref "./setup-queue-advanced.md" >}}) をご参照ください。