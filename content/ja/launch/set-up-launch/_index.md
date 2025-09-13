---
title: Launch のセットアップ
menu:
  launch:
    identifier: ja-launch-set-up-launch-_index
    parent: launch
weight: 3
---

このページでは、W&B Launch をセットアップするための大まかな手順を説明します:
1. 1. **キューをセットアップ**: キューは FIFO で、キュー設定を持ちます。キューの設定は、ジョブをどのターゲット リソースで、どのように実行するかを制御します。
2. **エージェントをセットアップ**: エージェントはあなたのマシン / インフラストラクチャー上で動作し、1 つ以上のキューをポーリングして Launch ジョブを取得します。ジョブを取得すると、エージェントはイメージがビルド済みで利用可能であることを確認します。その後、エージェントはジョブをターゲット リソースに送信します。

## キューをセットアップ
Launch のキューは、対象とする特定のターゲット リソースを指すように設定し、そのリソース固有の追加の設定も指定する必要があります。たとえば、Kubernetes クラスターを指す Launch キューでは、環境変数を含めたり、キュー設定でカスタムの名前空間を設定したりできます。キューを作成する際には、使用したいターゲット リソースと、そのリソースで使用する設定の両方を指定します。
エージェントがキューからジョブを受け取ると、キュー設定も受け取ります。エージェントがジョブをターゲット リソースに送信するときは、ジョブ側での上書きがある場合でも、キュー設定を含めて送信します。たとえば、ジョブの設定で、そのジョブ インスタンスに限り Amazon SageMaker のインスタンスタイプを指定できます。この場合、エンドユーザー向けのインターフェースとして [queue config templates]({{< relref path="./setup-queue-advanced.md#configure-queue-template" lang="ja" >}}) を使うのが一般的です。 

### キューを作成
1. [wandb.ai/launch](https://wandb.ai/launch) の Launch App に移動します。 
2. 画面右上の **create queue** ボタンをクリックします。 
{{< img src="/images/launch/create-queue.gif" alt="Launch の キューを作成" >}}
3. **Entity** のドロップダウン メニューから、このキューが属する Entity を選択します。 
4. **Queue** フィールドにキュー名を入力します。 
5. **Resource** のドロップダウンから、このキューに追加されたジョブが使用するコンピュート リソースを選択します。
6. このキューで **Prioritization** を許可するかどうかを選びます。優先度付けを有効にすると、Team の ユーザー がジョブをエンキューする際に Launch ジョブの優先度を設定できます。優先度の高いジョブは、低いジョブより先に実行されます。
7. **Configuration** フィールドに、JSON または YAML 形式でリソース設定を記入します。設定ドキュメントの構造と意味は、キューが指すリソースの種類に依存します。詳細は、対象リソースごとのセットアップ ページを参照してください。

## Launch エージェントをセットアップ
Launch エージェントは、1 つ以上の Launch キューをジョブのためにポーリングする長時間動作の プロセス です。Launch エージェントは、参照しているキューに応じて、FIFO もしくは優先度順でジョブをデキューします。エージェントがキューからジョブをデキューすると、必要に応じてそのジョブのイメージをビルドします。その後、エージェントはキュー設定で指定されたオプションとともにジョブをターゲット リソースに送信します。
{{% alert %}}
エージェントは非常に柔軟で、幅広い ユースケース に対応するように設定できます。必要な設定は、あなたの ユースケース によって異なります。詳しくは、各専用ページ（[Docker]({{< relref path="./setup-launch-docker.md" lang="ja" >}})、[Amazon SageMaker]({{< relref path="./setup-launch-sagemaker.md" lang="ja" >}})、[Kubernetes]({{< relref path="./setup-launch-kubernetes.md" lang="ja" >}})、[Vertex AI]({{< relref path="./setup-vertex.md" lang="ja" >}})）を参照してください。
{{% /alert %}}
{{% alert %}}
W&B は、特定の ユーザー の APIキー ではなく、サービス アカウントの APIキー でエージェントを起動することを推奨します。サービス アカウントの APIキー を使う利点は 2 つあります:
1. エージェントが個々の ユーザー に依存しません。
2. Launch を通じて作成された run に関連付けられる作成者は、エージェントに紐づく ユーザー ではなく、Launch によってその Launch ジョブを送信した ユーザー として扱われます。
{{% /alert %}}

### エージェントの設定
`launch-config.yaml` という名前の YAML ファイルで Launch エージェントを 設定 します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` に設定ファイルがあるか確認します。Launch エージェントを起動するときに、別の ディレクトリー を指定することもできます。
Launch エージェントの設定ファイルの内容は、エージェントの 環境、Launch キューのターゲット リソース、Docker ビルダー要件、クラウド レジストリ要件などによって異なります。 
ユースケース に関わらず、Launch エージェントには中核となる設定オプションがあります:
* `max_jobs`: エージェントが並列に実行できるジョブの最大数 
* `entity`: キューが属している Entity
* `queues`: エージェントが監視する 1 つ以上のキュー名
{{% alert %}}
W&B CLI を使って（設定 YAML ファイルの代わりに）Launch エージェントの共通オプションを指定できます: 最大ジョブ数、W&B の Entity、Launch キュー。詳細は [`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドを参照してください。
{{% /alert %}}
次の YAML の コードスニペット は、Launch エージェントの基本的な設定キーを指定する方法を示しています:
```yaml title="launch-config.yaml"
# 同時に実行する Runs の最大数。-1 = 無制限
max_jobs: -1

entity: <entity-name>

# ポーリングするキューの一覧。
queues:
  - <queue-name>
```

### コンテナー ビルダーを設定
Launch エージェントはイメージをビルドするように 設定 できます。Git リポジトリや code Artifacts から作成された Launch ジョブを使う場合は、コンテナー ビルダーを使うようにエージェントを設定する必要があります。Launch ジョブの作成方法については、[Create a launch job]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) を参照してください。 
W&B Launch は 3 つのビルダー オプションをサポートします:
* Docker: ローカルの Docker デーモンを使ってイメージをビルドします。
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Docker デーモンが利用できない環境でもイメージをビルドできる、Google のプロジェクトです。 
* Noop: エージェントはジョブのビルドを行わず、事前にビルドされたイメージのみを pull します。
{{% alert %}}
Docker デーモンが利用できない 環境（例: Kubernetes クラスター）でエージェントがポーリングする場合は、Kaniko ビルダーを使用してください。
Kaniko ビルダーの詳細は [Set up Kubernetes]({{< relref path="./setup-launch-kubernetes.md" lang="ja" >}}) を参照してください。
{{% /alert %}}
イメージ ビルダーを指定するには、エージェントの設定に builder キーを含めます。たとえば、次の コードスニペット は Docker または Kaniko を使用するよう指定した launch 設定（`launch-config.yaml`）の一部です:
```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### コンテナー レジストリを設定
場合によっては、Launch エージェントを クラウド レジストリに接続したいことがあります。よくあるシナリオは次のとおりです:
* ビルドした場所以外の 環境（高性能なワークステーションや クラスター など）でジョブを実行したい。
* エージェントでイメージをビルドし、そのイメージを Amazon SageMaker や Vertex AI で実行したい。
* エージェントに、イメージ リポジトリから pull するための認証情報を提供させたい。
コンテナー レジストリと連携するようエージェントを設定する方法の詳細は、[Advanced agent set]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) ページを参照してください。

## Launch エージェントを起動
`launch-agent` の W&B CLI コマンドで Launch エージェントを起動します:
```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```
ユースケース によっては、Kubernetes クラスター内からキューをポーリングする Launch エージェントを動かしたいことがあります。詳しくは [Advanced queue set up page]({{< relref path="./setup-queue-advanced.md" lang="ja" >}}) を参照してください。