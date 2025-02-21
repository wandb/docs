---
title: Set up Launch
menu:
  launch:
    identifier: ja-launch-set-up-launch-_index
    parent: launch
weight: 3
---

このページでは、W&B Launch の設定に必要な大まかな手順について説明します。

1.  **キューの設定**: キューは FIFO (先入れ先出し) であり、キューの設定を持っています。キューの設定は、ターゲットリソース上でジョブがどこでどのように実行されるかを制御します。
2.  **エージェントの設定**: エージェントは、あなたのマシン/インフラストラクチャー上で実行され、Launch ジョブのために 1 つまたは複数のキューをポーリングします。ジョブがプルされると、エージェントはイメージがビルドされ、利用可能であることを確認します。その後、エージェントはジョブをターゲットリソースに送信します。

## キューの設定
Launch キューは、特定ターゲットリソースと、そのリソースに固有の追加設定を指すように設定する必要があります。たとえば、Kubernetes クラスターを指す Launch キューには、環境変数を含めたり、Launch キュー設定でカスタム名前空間を設定したりする場合があります。キューを作成する際には、使用するターゲットリソースと、使用するリソースの設定の両方を指定します。

エージェントがキューからジョブを受信すると、キュー設定も受信します。エージェントがジョブをターゲットリソースに送信する際、ジョブ自体からの上書きとともに、キュー設定が含まれます。たとえば、ジョブ設定を使用して、そのジョブインスタンスのみの Amazon SageMaker インスタンスタイプを指定できます。この場合、エンド ユーザーインターフェイスとして[キュー設定テンプレート]({{< relref path="./setup-queue-advanced.md#configure-queue-template" lang="ja" >}})を使用するのが一般的です。

### キューの作成
1.  [wandb.ai/launch](https://wandb.ai/launch) の Launch アプリに移動します。
2.  画面の右上にある **キューの作成** ボタンをクリックします。

{{< img src="/images/launch/create-queue.gif" alt="" >}}

3.  **Entity** ドロップダウンメニューから、キューが属する Entity を選択します。
4.  **Queue** フィールドに、キューの名前を入力します。
5.  **Resource** ドロップダウンから、このキューに追加されたジョブに使用するコンピューティングリソースを選択します。
6.  このキューで **Prioritization** を許可するかどうかを選択します。優先順位付けが有効になっている場合、チームのユーザーは、エンキュー時に Launch ジョブの優先順位を定義できます。優先度の高いジョブは、優先度の低いジョブよりも先に実行されます。
7.  **Configuration** フィールドに、JSON または YAML 形式でリソース設定を入力します。設定ドキュメントの構造とセマンティクスは、キューが指しているリソースタイプによって異なります。詳細については、ターゲットリソースの専用設定ページを参照してください。

## Launch エージェントの設定
Launch エージェントは、ジョブのために 1 つまたは複数の Launch キューをポーリングする長時間実行プロセスです。Launch エージェントは、先入れ先出し (FIFO) 順、またはポーリング元のキューに応じて優先順位順にジョブをデキューします。エージェントがキューからジョブをデキューすると、オプションでそのジョブのイメージを構築します。その後、エージェントはジョブをターゲットリソースに、キュー設定で指定された設定オプションとともに送信します。

{{% alert %}}
エージェントは非常に柔軟性があり、さまざまなユースケースをサポートするように設定できます。エージェントに必要な設定は、特定のユースケースによって異なります。 [Docker]({{< relref path="./setup-launch-docker.md" lang="ja" >}}), [Amazon SageMaker]({{< relref path="./setup-launch-sagemaker.md" lang="ja" >}}), [Kubernetes]({{< relref path="./setup-launch-kubernetes.md" lang="ja" >}}), または [Vertex AI]({{< relref path="./setup-vertex.md" lang="ja" >}}) の専用ページを参照してください。
{{% /alert %}}

{{% alert %}}
W&B では、特定 User の APIキー ではなく、サービスアカウントの APIキー でエージェントを開始することをお勧めします。サービスアカウントの APIキー を使用することには、次の 2 つの利点があります。
1.  エージェントは個々の User に依存しません。
2.  Launch を介して作成された Run に関連付けられた作成者は、エージェントに関連付けられた User ではなく、Launch ジョブを送信した User として Launch によって認識されます。
{{% /alert %}}

### エージェントの設定
`launch-config.yaml` という YAML ファイルで Launch エージェントを設定します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` にある設定ファイルを確認します。Launch エージェントをアクティブ化する際に、別のディレクトリーをオプションで指定できます。

Launch エージェントの設定ファイルの内容は、Launch エージェントの環境、Launch キューのターゲットリソース、Docker ビルダーの要件、クラウドレジストリの要件などによって異なります。

ユースケースに関係なく、Launch エージェントには、構成可能な主要オプションがあります。
*   `max_jobs`: エージェントが並行して実行できるジョブの最大数
*   `entity`: キューが属する Entity
*   `queues`: エージェントが監視する 1 つまたは複数のキューの名前

{{% alert %}}
W&B CLI を使用して、Launch エージェントの汎用的な構成可能オプション (設定 YAML ファイルの代わりに): ジョブの最大数、W&B Entity、および Launch キューを指定できます。詳細については、[`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドを参照してください。
{{% /alert %}}

次の YAML コードスニペットは、主要な Launch エージェントの設定 Key を指定する方法を示しています。

```yaml title="launch-config.yaml"
# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

entity: <entity-name>

# List of queues to poll.
queues:
  - <queue-name>
```

### コンテナビルダーの設定
Launch エージェントは、イメージを構築するように設定できます。git リポジトリーまたは コード Artifacts から作成された Launch ジョブを使用する場合は、コンテナビルダーを使用するようにエージェントを設定する必要があります。 Launch ジョブの作成方法の詳細については、[Launch ジョブの作成]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ja" >}})を参照してください。

W&B Launch は、次の 3 つのビルダーオプションをサポートしています。

*   Docker: Docker ビルダーは、ローカルの Docker デーモンを使用してイメージを構築します。
*   [Kaniko](https://github.com/GoogleContainerTools/kaniko): Kaniko は、Docker デーモンが利用できない環境でイメージを構築できる Google プロジェクトです。
*   Noop: エージェントはジョブを構築しようとせず、代わりに事前に構築されたイメージのみをプルします。

{{% alert %}}
エージェントが Docker デーモンが利用できない環境 (たとえば、Kubernetes クラスター) でポーリングしている場合は、Kaniko ビルダーを使用します。

Kaniko ビルダーの詳細については、[Kubernetes の設定]({{< relref path="./setup-launch-kubernetes.md" lang="ja" >}})を参照してください。
{{% /alert %}}

イメージビルダーを指定するには、エージェント設定に builder Key を含めます。たとえば、次のコードスニペットは、Docker または Kaniko を使用するように指定する Launch 設定 (`launch-config.yaml`) の一部を示しています。

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### コンテナレジストリーの設定
場合によっては、Launch エージェントをクラウドリレジストリーに接続することが必要な場合があります。 Launch エージェントをクラウドリレジストリーに接続することが必要な一般的なシナリオを次に示します。

*   強力なワークステーションやクラスターなど、構築した環境とは別の環境でジョブを実行したい。
*   エージェントを使用してイメージを構築し、これらのイメージを Amazon SageMaker または VertexAI で実行したい。
*   Launch エージェントに、イメージリポジトリーからプルするための認証情報を提供させたい。

エージェントがコンテナレジストリーとやり取りするように設定する方法の詳細については、[エージェントの詳細設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}})ページを参照してください。

## Launch エージェントのアクティブ化
`launch-agent` W&B CLI コマンドで Launch エージェントをアクティブ化します。

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

一部のユースケースでは、Kubernetes クラスター内から Launch エージェントにキューをポーリングさせることが必要な場合があります。詳細については、[キューの詳細設定ページ]({{< relref path="./setup-queue-advanced.md" lang="ja" >}})を参照してください。
