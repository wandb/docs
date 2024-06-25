---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Set up Launch

このページでは W&B Launch をセットアップするための高レベルの手順について説明します:

1. **キューをセットアップ**: キューは FIFO であり、キューの設定を持っています。キューの設定は、ターゲットリソースでジョブがどのように実行されるかを制御します。
2. **エージェントをセットアップ**: エージェントはマシン/インフラストラクチャー上で動作し、キューをポーリングして launch ジョブを取得します。ジョブがプルされると、エージェントはそのイメージがビルドされ利用可能であることを確認します。エージェントはその後、ジョブをターゲットリソースに提出します。

## キューをセットアップ
Launch キューは、特定のターゲットリソースに向けて設定される必要があります。また、そのリソースに固有の追加設定も必要です。たとえば、Kubernetes クラスターを指す launch キューの場合、環境変数を含めるか、カスタムネームスペースを設定することがあります。キューを作成する際には、使用したいターゲットリソースと、そのリソース用の設定を指定します。

エージェントがキューからジョブを受け取ると、キューの設定も受け取ります。エージェントがジョブをターゲットリソースに提出するときには、キューの設定と、ジョブ自体からの上書き設定を含めます。たとえば、ジョブ設定を使用して特定のジョブインスタンスだけの Amazon SageMaker インスタンスタイプを指定することができます。この場合、[queue config templates](./setup-queue-advanced.md#configure-queue-template) をエンドユーザーインターフェースとして使用するのが一般的です。

### キューを作成
1. [wandb.ai/launch](https://wandb.ai/launch) で Launch App に移動します。
2. 画面右上の **create queue** ボタンをクリックします。

![](/images/launch/create-queue.gif)

3. **Entity** ドロップダウンメニューから、キューが属するエンティティを選択します。
  :::tip
  チームエンティティを選択すると、チームの全メンバーがこのキューにジョブを送信できるようになります。個人のエンティティ（ユーザー名に関連付けられている場合）を選択すると、W&B はそのユーザーだけが使用できるプライベートキューを作成します。
  :::
4. **Queue** フィールドにキューの名前を入力します。
5. **Resource** ドロップダウンから、このキューにジョブを追加するコンピュートリソースを選択します。
6. **Prioritization** を許可するかどうかを選択します。優先順位付けが有効になっている場合、チームのユーザーがジョブをキューに追加する際に優先順位を設定できます。高い優先順位のジョブが低い優先順位のジョブよりも先に実行されます。
7. **Configuration** フィールドに JSON または YAML 形式でリソース設定を提供します。設定ドキュメントの構造と意味は、キューが指すリソースタイプに依存します。詳細については、ターゲットリソースの専用設定ページを参照してください。

## launch エージェントをセットアップ
Launch エージェントは、ジョブのために複数の launch キューをポーリングする長時間稼働のプロセスです。Launch エージェントは、最初に入力されたジョブから実行するか、優先順位順に従ってジョブをデキューします。エージェントがキューからジョブをデキューすると、そのジョブ用のイメージをビルドするオプションがあります。その後、エージェントはターゲットリソースに設定オプションを含めてジョブを提出します。

:::INFO
エージェントは非常に柔軟で、多様なユースケースをサポートするように設定できます。エージェントに必要な設定は、特定のユースケースに依存します。詳しくは [Docker](./setup-launch-docker.md)、 [Amazon SageMaker](./setup-launch-sagemaker.md)、 [Kubernetes](./setup-launch-kubernetes.md) または [Vertex AI](./setup-vertex.md) の専用ページをご覧ください。
:::

:::tip
W&B では、エージェントを特定のユーザーの API キーではなく[サービスアカウント](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful)の API キーで起動することを推奨しています。これは次の2つのメリットを持ちます:
1. エージェントは個々のユーザーに依存しません。
2. Launch を通じて作成された run に関連付けられた著者が、エージェントに関連付けられたユーザーではなく、launch ジョブを提出したユーザーとして Launch に表示されます。
:::

### エージェント設定
launch エージェントは `launch-config.yaml` と呼ばれる YAML ファイルで設定します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` に設定ファイルを確認します。launch エージェントを起動するときに異なるディレクトリーを指定することもできます。

launch エージェントの設定ファイルの内容は、launch エージェントの環境、launch キューのターゲットリソース、Docker ビルダーの要件、クラウドレジストリの要件などに依存します。

ユースケースに関係なく、launch エージェントには基本的な設定オプションがあります:
* `max_jobs`: エージェントが並行して実行できる最大ジョブ数
* `entity`: キューが所属するエンティティ
* `queues`: エージェントが監視する1つ以上のキューの名前

:::tip
W&B CLI を使用して、launch エージェントのための普遍的な設定オプション（設定 YAML ファイルの代わりに）を指定することができます: 最大ジョブ数、W&B エンティティ、および launch キュー。詳細は [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) コマンドを参照してください。
:::

次の YAML のスニペットは、基本的な launch エージェント設定キーを指定する方法を示しています:

```yaml title="launch-config.yaml"
# Max number of concurrent runs to perform. -1 = no limit
max_jobs: -1

entity: <entity-name>

# List of queues to poll.
queues:
  - <queue-name>
```

### コンテナビルダーの設定
launch エージェントはイメージをビルドするように設定できます。Git リポジトリやコードアーティファクトから作成された launch ジョブを使用する場合、エージェントにコンテナビルダーを使用するように設定する必要があります。Launch ジョブを作成する方法の詳細については [Create a launch job](./create-launch-job.md) を参照してください。

W&B Launch は次の3つのビルダーオプションをサポートしています:
* Docker: Dockerビルダーはローカルの Docker デーモンを使用してイメージをビルドします。
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Kaniko は、Docker デーモンが利用できない環境でイメージをビルドできる Google プロジェクトです。
* Noop: エージェントはジョブのビルドを試みず、事前ビルド済みのイメージのみをプルします。

:::tip
エージェントが Docker デーモンが利用できない環境（例: Kubernetes クラスター）でポーリングしている場合は、Kaniko ビルダーを使用してください。

Kaniko ビルダーの詳細については [Set up Kubernetes](./setup-launch-kubernetes.md) を参照してください。
:::

イメージビルダーを指定するには、エージェントの設定にビルダーキーを含めます。たとえば、次のコードスニペットは Docker または Kaniko を使用することを指定する launch 設定（`launch-config.yaml`）の一部を示しています:

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### コンテナレジストリの設定
場合によっては、launch エージェントをクラウドレジストリに接続することが望ましいことがあります。launch エージェントをクラウドレジストリに接続したい一般的なシナリオには次のものがあります:

* ビルドした場所とは異なる環境（強力な作業ステーションやクラスターなど）でジョブを実行したい場合。
* エージェントを使用してイメージをビルドし、これらのイメージを Amazon SageMaker や VertexAI で実行したい場合。
* launch エージェントがイメージレポジトリからプルするための認証情報を提供することを希望する場合。

エージェントをコンテナレジストリとの連携設定する方法の詳細については、[Advanced agent set](./setup-agent-advanced.md)の設定ページを参照してください。

## launch エージェントを起動
launch エージェントを起動するには、`launch-agent` W&B CLI コマンドを使用します:

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

いくつかのユースケースでは、Kubernetes クラスター内からキューをポーリングする launch エージェントを持つことが望ましい場合があります。詳細については、[Advanced queue set up page](./setup-queue-advanced.md) を参照してください。