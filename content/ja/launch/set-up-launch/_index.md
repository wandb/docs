---
title: Set up Launch
menu:
  launch:
    identifier: ja-launch-set-up-launch-_index
    parent: launch
weight: 3
---

このページでは、W&B Launch を設定するために必要な大まかな手順について説明します。

1. **キューの設定**: キューは FIFO であり、キュー設定を備えています。キューの設定は、ターゲットリソース上でジョブがどこでどのように実行されるかを制御します。
2. **エージェントの設定**: エージェントは、ユーザーのマシン/インフラストラクチャー上で実行され、Launch ジョブの 1 つ以上のキューをポーリングします。ジョブがプルされると、エージェントはイメージが構築され、利用可能であることを確認します。その後、エージェントはジョブをターゲットリソースに送信します。

## キューの設定
Launch キューは、特定ターゲットリソースと、そのリソースに固有の追加設定を指すように設定する必要があります。たとえば、Kubernetes クラスターを指す Launch キューには、環境変数を含めたり、Launch キュー設定のカスタム名前空間を設定したりできます。キューを作成する際には、使用するターゲットリソースと、そのリソースが使用する設定の両方を指定します。

エージェントがキューからジョブを受信すると、キュー設定も受信します。エージェントがジョブをターゲットリソースに送信する際、ジョブ自体のオーバーライドとともにキュー設定が含まれます。たとえば、ジョブ設定を使用して、そのジョブインスタンスのみの Amazon SageMaker インスタンスタイプを指定できます。この場合、[キュー設定テンプレート]({{< relref path="./setup-queue-advanced.md#configure-queue-template" lang="ja" >}})をエンドユーザーインターフェイスとして使用するのが一般的です。

### キューの作成
1. [wandb.ai/launch](https://wandb.ai/launch) で Launch アプリケーションに移動します。
2. 画面右上の **create queue** ボタンをクリックします。

{{< img src="/images/launch/create-queue.gif" alt="" >}}

3. **Entity** ドロップダウンメニューから、キューが属するエンティティを選択します。
4. **Queue** フィールドにキューの名前を入力します。
5. **Resource** ドロップダウンから、このキューに追加されたジョブで使用するコンピュートリソースを選択します。
6. このキューの **Prioritization** を許可するかどうかを選択します。優先順位付けが有効になっている場合、チームのユーザーは、エンキュー時に Launch ジョブの優先順位を定義できます。優先度の高いジョブは、優先度の低いジョブよりも先に実行されます。
7. **Configuration** フィールドに、JSON または YAML 形式でリソース設定を入力します。設定ドキュメントの構造とセマンティクスは、キューが指すリソースタイプによって異なります。詳細については、ターゲットリソースの専用設定ページを参照してください。

## Launch エージェントの設定
Launch エージェントは、ジョブのために 1 つ以上の Launch キューをポーリングする、長時間実行されるプロセスです。Launch エージェントは、先入れ先出し（FIFO）順、またはプル元のキューに応じて優先順位順にジョブをデキューします。エージェントがキューからジョブをデキューすると、オプションでそのジョブのイメージを構築します。その後、エージェントはジョブをターゲットリソースに、キュー設定で指定された設定オプションととも​​に送信します。

{{% alert %}}
エージェントは非常に柔軟性があり、さまざまなユースケースをサポートするように構成できます。エージェントに必要な設定は、特定のユースケースによって異なります。 [Docker]({{< relref path="./setup-launch-docker.md" lang="ja" >}}), [Amazon SageMaker]({{< relref path="./setup-launch-sagemaker.md" lang="ja" >}}), [Kubernetes]({{< relref path="./setup-launch-kubernetes.md" lang="ja" >}}), または [Vertex AI]({{< relref path="./setup-vertex.md" lang="ja" >}}) の専用ページを参照してください。
{{% /alert %}}

{{% alert %}}
W&B では、特定ユーザーの API キーではなく、サービスアカウントの API キーでエージェントを開始することをお勧めします。サービスアカウントの API キーを使用することには、次の 2 つの利点があります。
1. エージェントは、個々のユーザーに依存しません。
2. Launch を介して作成された run に関連付けられた作成者は、エージェントに関連付けられたユーザーではなく、Launch ジョブを送信したユーザーとして Launch によって認識されます。
{{% /alert %}}

### エージェントの設定
`launch-config.yaml` という YAML ファイルで Launch エージェントを設定します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` にある設定ファイルを確認します。Launch エージェントをアクティブ化するときに、別のディレクトリーをオプションで指定できます。

Launch エージェントの設定ファイルの内容は、Launch エージェントの環境、Launch キューのターゲットリソース、Docker ビルダーの要件、クラウドリポジトリの要件などによって異なります。

ユースケースに関係なく、Launch エージェントには、設定可能な主要オプションがあります。
* `max_jobs`: エージェントが並行して実行できるジョブの最大数
* `entity`: キューが属するエンティティ
* `queues`: エージェントが監視する 1 つ以上のキューの名前

{{% alert %}}
W&B CLI を使用して、Launch エージェントの普遍的な設定可能オプション（設定 YAML ファイルの代わりに）、ジョブの最大数、W&B エンティティ、および Launch キューを指定できます。詳細については、[`wandb launch-agent`]({{< relref path="/ref/cli/wandb-launch-agent.md" lang="ja" >}}) コマンドを参照してください。
{{% /alert %}}

次の YAML コードスニペットは、主要な Launch エージェント設定キーを指定する方法を示しています。

```yaml title="launch-config.yaml"
# 実行する同時runsの最大数。 -1 = 無制限
max_jobs: -1

entity: <entity-name>

# ポーリングするキューのリスト。
queues:
  - <queue-name>
```

### コンテナビルダーの設定
Launch エージェントは、イメージを構築するように構成できます。git リポジトリまたはコード Artifacts から作成された Launch ジョブを使用する場合は、コンテナビルダーを使用するようにエージェントを設定する必要があります。 Launch ジョブの作成方法の詳細については、[Launch ジョブの作成]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ja" >}})を参照してください。

W&B Launch は、次の 3 つのビルダーオプションをサポートしています。

* Docker: Docker ビルダーは、ローカル Docker デーモンを使用してイメージを構築します。
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Kaniko は、Docker デーモンが利用できない環境でイメージを構築できる Google プロジェクトです。
* Noop: エージェントはジョブの構築を試行せず、代わりに構築済みのイメージのみをプルします。

{{% alert %}}
エージェントが Docker デーモンが利用できない環境（Kubernetes クラスターなど）でポーリングしている場合は、Kaniko ビルダーを使用してください。

Kaniko ビルダーの詳細については、[Kubernetes の設定]({{< relref path="./setup-launch-kubernetes.md" lang="ja" >}})を参照してください。
{{% /alert %}}

イメージビルダーを指定するには、エージェント設定に builder キーを含めます。たとえば、次のコードスニペットは、Docker または Kaniko を使用するように指定する Launch 設定（`launch-config.yaml`）の一部を示しています。

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### コンテナレジストリの設定
場合によっては、Launch エージェントをクラウドリポジトリに接続する必要があるかもしれません。Launch エージェントをクラウドリポジトリに接続する一般的なシナリオとしては、次のようなものがあります。

* 強力なワークステーションやクラスターなど、イメージを構築したのとは別の環境でジョブを実行する場合。
* エージェントを使用してイメージを構築し、これらのイメージを Amazon SageMaker または VertexAI で実行する場合。
* Launch エージェントに、イメージリポジトリからプルするための認証情報を提供させる場合。

コンテナレジストリとやり取りするようにエージェントを設定する方法の詳細については、[エージェントの詳細設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}})ページを参照してください。

## Launch エージェントのアクティブ化
`launch-agent` W&B CLI コマンドで Launch エージェントをアクティブ化します。

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

一部のユースケースでは、Kubernetes クラスター内から Launch エージェントにキューをポーリングさせたい場合があります。詳細については、[キューの詳細設定ページ]({{< relref path="./setup-queue-advanced.md" lang="ja" >}})を参照してください。
