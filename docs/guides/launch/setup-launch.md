---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Set up Launch

このページでは、W&B Launchをセットアップするための高レベルな手順を説明します。

1. **キューの設定**: キューはFIFOであり、キュー設定を持ちます。キューの設定は、ジョブがターゲットリソースでどのように実行されるかを制御します。
2. **エージェントの設定**: エージェントはあなたのマシンやインフラストラクチャー上で動作し、1つ以上のキューからlaunchジョブをポーリングします。ジョブが取得されると、エージェントはイメージがビルドされ利用可能であることを確認します。その後、エージェントはジョブをターゲットリソースに送信します。

## キューの設定
Launchキューは、特定のターゲットリソースを指すように設定する必要があります。また、そのリソースに特有の追加設定も必要です。例えば、Kubernetesクラスターを指すlaunchキューには、環境変数やカスタムネームスペースの設定が含まれることがあります。キューを作成する際には、使用したいターゲットリソースとそのリソースの設定を指定します。

エージェントがキューからジョブを受け取ると、キュー設定も受け取ります。エージェントがジョブをターゲットリソースに送信する際には、キュー設定とジョブ自体のオーバーライドを含めます。例えば、ジョブ設定を使用して、そのジョブインスタンス専用のAmazon SageMakerインスタンスタイプを指定することができます。この場合、エンドユーザーインターフェースとして[queue config templates](./setup-queue-advanced.md#configure-queue-template)を使用することが一般的です。

### キューの作成
1. [wandb.ai/launch](https://wandb.ai/launch)でLaunchアプリに移動します。
2. 画面右上の**create queue**ボタンをクリックします。

![](/images/launch/create-queue.gif)

3. **Entity**ドロップダウンメニューから、キューが属するエンティティを選択します。
  :::tip
  チームエンティティを選択すると、チームの全メンバーがこのキューにジョブを送信できるようになります。個人エンティティ（ユーザー名に関連付けられたもの）を選択すると、W&Bはそのユーザーのみが使用できるプライベートキューを作成します。
  :::
4. **Queue**フィールドにキューの名前を入力します。
5. **Resource**ドロップダウンから、このキューに追加されるジョブが使用するコンピュートリソースを選択します。
6. このキューに対して**Prioritization**を許可するかどうかを選択します。優先順位付けが有効な場合、チームのユーザーはジョブをキューに追加する際に優先順位を設定できます。高い優先順位のジョブは低い優先順位のジョブよりも先に実行されます。
7. **Configuration**フィールドにJSONまたはYAML形式でリソース設定を入力します。設定ドキュメントの構造と意味は、キューが指すリソースタイプによって異なります。詳細は、ターゲットリソースの専用セットアップページを参照してください。

## launchエージェントの設定
Launchエージェントは、1つ以上のlaunchキューからジョブをポーリングする長時間実行プロセスです。Launchエージェントは、FIFO順または優先順位順にジョブをデキューします。エージェントがキューからジョブをデキューすると、そのジョブのイメージをオプションでビルドします。その後、エージェントはキュー設定に指定された設定オプションと共にジョブをターゲットリソースに送信します。

:::info
エージェントは非常に柔軟で、多様なユースケースをサポートするように設定できます。エージェントに必要な設定は、特定のユースケースによって異なります。詳細は、[Docker](./setup-launch-docker.md)、[Amazon SageMaker](./setup-launch-sagemaker.md)、[Kubernetes](./setup-launch-kubernetes.md)、または[Vertex AI](./setup-vertex.md)の専用ページを参照してください。
:::

:::tip
W&Bは、特定のユーザーのAPIキーではなく、[サービスアカウント](https://docs.wandb.ai/guides/technical-faq/general#what-is-a-service-account-and-why-is-it-useful)のAPIキーでエージェントを開始することを推奨します。サービスアカウントのAPIキーを使用する利点は2つあります。
1. エージェントが個々のユーザーに依存しません。
2. Launchを通じて作成されたrunに関連付けられた著者は、エージェントに関連付けられたユーザーではなく、launchジョブを送信したユーザーとして表示されます。
:::

### エージェント設定
launchエージェントは、`launch-config.yaml`という名前のYAMLファイルで設定します。デフォルトでは、W&Bは`~/.config/wandb/launch-config.yaml`に設定ファイルをチェックします。launchエージェントをアクティブにする際に、別のディレクトリーを指定することもできます。

launchエージェントの設定ファイルの内容は、エージェントの環境、launchキューのターゲットリソース、Dockerビルダーの要件、クラウドレジストリの要件などに依存します。

ユースケースに関係なく、launchエージェントにはコア設定オプションがあります。
* `max_jobs`: エージェントが並行して実行できる最大ジョブ数
* `entity`: キューが属するエンティティ
* `queues`: エージェントが監視する1つ以上のキューの名前

:::tip
W&B CLIを使用して、launchエージェントのユニバーサル設定オプション（設定YAMLファイルの代わりに）を指定できます。最大ジョブ数、W&Bエンティティ、およびlaunchキューです。詳細は[`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md)コマンドを参照してください。
:::

以下のYAMLスニペットは、コアlaunchエージェント設定キーを指定する方法を示しています。

```yaml title="launch-config.yaml"
# 実行する同時runの最大数。-1 = 制限なし
max_jobs: -1

entity: <entity-name>

# ポーリングするキューのリスト
queues:
  - <queue-name>
```

### コンテナビルダーの設定
launchエージェントはイメージをビルドするように設定できます。gitリポジトリやコードアーティファクトから作成されたlaunchジョブを使用する場合、エージェントをコンテナビルダーを使用するように設定する必要があります。launchジョブの作成方法については、[Create a launch job](./create-launch-job.md)を参照してください。

W&B Launchは3つのビルダーオプションをサポートしています。
* Docker: DockerビルダーはローカルのDockerデーモンを使用してイメージをビルドします。
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): Kanikoは、Dockerデーモンが利用できない環境でイメージをビルドするためのGoogleプロジェクトです。
* Noop: エージェントはジョブをビルドせず、事前にビルドされたイメージのみをプルします。

:::tip
エージェントがDockerデーモンが利用できない環境（例えば、Kubernetesクラスター）でポーリングしている場合は、Kanikoビルダーを使用してください。

Kanikoビルダーの詳細については、[Set up Kubernetes](./setup-launch-kubernetes.md)を参照してください。
:::

イメージビルダーを指定するには、エージェント設定にビルダーキーを含めます。例えば、以下のコードスニペットは、DockerまたはKanikoを使用するように指定するlaunch設定（`launch-config.yaml`）の一部を示しています。

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### コンテナレジストリの設定
場合によっては、launchエージェントをクラウドレジストリに接続したいことがあります。launchエージェントをクラウドレジストリに接続したい一般的なシナリオには以下が含まれます。

* ビルドした場所とは異なる環境（例えば、強力なワークステーションやクラスター）でジョブを実行したい場合。
* エージェントを使用してイメージをビルドし、これらのイメージをAmazon SageMakerやVertexAIで実行したい場合。
* launchエージェントがイメージリポジトリからプルするための認証情報を提供する必要がある場合。

エージェントをコンテナレジストリと連携させる方法については、[Advanced agent set](./setup-agent-advanced.md)アップページを参照してください。

## launchエージェントのアクティベート
`launch-agent` W&B CLIコマンドを使用してlaunchエージェントをアクティブにします。

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

ユースケースによっては、Kubernetesクラスター内からキューをポーリングするlaunchエージェントを持ちたい場合があります。詳細は[Advanced queue set up page](./setup-queue-advanced.md)を参照してください。