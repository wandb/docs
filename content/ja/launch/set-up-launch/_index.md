---
title: Set up Launch
menu:
  launch:
    identifier: ja-launch-set-up-launch-_index
    parent: launch
weight: 3
---

このページでは、W&B Launchを設定するための高レベルな手順を説明します。

1. **キューを設定する**: キューはFIFOであり、キュー設定を持っています。キューの設定は、ジョブがターゲットリソースでどのように実行されるかを制御します。
2. **エージェントを設定する**: エージェントは、マシン/インフラストラクチャー上で動作し、1つ以上のキューからLaunchジョブをポーリングします。ジョブがプルされると、エージェントはイメージが構築され、利用可能であることを確認します。その後、エージェントはジョブをターゲットリソースに送信します。

## キューを設定する
Launchキューは、特定のターゲットリソースと、そのリソースに特有の追加の設定を指すように設定されなければなりません。例えば、Kubernetesクラスターを指すLaunchキューには、環境変数やカスタムネームスペースの設定を含むことがあります。キューを作成するときには、使用したいターゲットリソースとそのリソースの設定を指定します。

エージェントがキューからジョブを受け取ると、キュー設定も受け取ります。エージェントがジョブをターゲットリソースに送信する際には、ジョブ自体からの上書きを含むキュー設定を含めます。例えば、ジョブ設定を使用して、そのジョブインスタンス専用のAmazon SageMakerインスタンスタイプを指定することができます。この場合、エンドユーザーインターフェースとして[キューコンフィグテンプレート](https://setup-queue-advanced.md#configure-queue-template)を使用することが一般的です。

### キューを作成する
1. [wandb.ai/launch](https://wandb.ai/launch)でLaunch Appに移動します。
2. 画面右上の**create queue**ボタンをクリックします。

{{< img src="/images/launch/create-queue.gif" alt="" >}}

3. **Entity** ドロップダウンメニューから、キューが属するエンティティを選択します。
4. **Queue** フィールドに、キューの名前を入力します。
5. **Resource** ドロップダウンから、このキューに追加したいコンピュートリソースを選択します。
6. このキューに対して**Prioritization**を許可するかどうかを選択します。優先度が有効な場合、チームのユーザーがキューにジョブを追加する際にそのジョブの優先度を定義できます。優先度の高いジョブは、優先度の低いジョブよりも先に実行されます。
7. **Configuration** フィールドには、リソースの設定をJSONまたはYAML形式で提供します。設定ドキュメントの構造とセマンティクスは、キューが指しているリソースタイプに依存します。詳細については、ターゲットリソースの専用設定ページを参照してください。

## Launchエージェントを設定する
Launchエージェントは、ジョブのために1つまたは複数のLaunchキューをポーリングする長時間実行されるプロセスです。Launchエージェントは、FIFO順または優先度順でジョブをキューから取り出します。エージェントがキューからジョブを取り出すと、そのジョブのためにオプションでイメージを構築します。その後、エージェントはキュー設定に指定された設定オプションとともに、そのジョブをターゲットリソースに送信します。

{{% alert %}}
エージェントは非常に柔軟で、多様なユースケースをサポートするように設定できます。エージェントの設定に必要な項目は、特定のユースケースに依存します。詳細については、[Docker](https://setup-launch-docker.md)、[Amazon SageMaker](https://setup-launch-sagemaker.md)、[Kubernetes](https://setup-launch-kubernetes.md)、または[Vertex AI](https://setup-vertex.md)の専用ページを参照してください。
{{% /alert %}}

{{% alert %}}
W&Bは、特定のユーザーのAPIキーではなく、サービスアカウントのAPIキーでエージェントを起動することをお勧めします。サービスアカウントのAPIキーを使用することには二つの利点があります：
1. エージェントが個々のユーザーに依存しない。
2. Launchを通じて作成されたrunに関連付けられた作成者は、エージェントに関連付けられたユーザーではなく、Launchジョブを送信したユーザーとしてLaunchによって認識される。
{{% /alert %}}

### エージェント設定
エージェントの設定は、`launch-config.yaml` という名前のYAMLファイルで行います。デフォルトでは、W&Bは`~/.config/wandb/launch-config.yaml`に設定ファイルを確認します。エージェントを起動する際に、異なるディレクトリーを指定することも可能です。

Launchエージェントの設定ファイルの内容は、Launchエージェントの環境、Launchキューのターゲットリソース、Dockerビルダーの要件、クラウドレジストリ要件などに依存します。

ユースケースに依存せず、Launchエージェントに対して設定可能なコアオプションがあります：
* `max_jobs`: エージェントが並行して実行できるジョブの最大数
* `entity`: キューが属するエンティティ
* `queues`: エージェントが監視する1つまたは複数のキューの名前

{{% alert %}}
W&B CLIを使用して、Launchエージェントのユニバーサル設定オプション（代わりにYAML設定ファイルに記載）を指定することが可能です：ジョブの最大数、W&Bエンティティ、およびLaunchキュー。詳しくは[`wandb launch-agent`](https://ref/cli/wandb-launch-agent.md) コマンドを参照してください。
{{% /alert %}}

以下のYAMLスニペットは、コアLaunchエージェント設定キーを指定する方法を示しています：

```yaml title="launch-config.yaml"
# 実行する並列runの最大数。-1 = 制限なし
max_jobs: -1

entity: <entity-name>

# ポーリングするキューのリスト。
queues:
  - <queue-name>
```

### コンテナービルダーを設定する
Launchエージェントはイメージを構築するために設定することができます。Gitリポジトリやコードアーティファクトから作成されたLaunchジョブを使用する場合、エージェントをコンテナービルダーを利用するように設定する必要があります。Launchジョブの作成方法について詳しくは[Create a launch job](../create-and-deploy-jobs/create-launch-job.md)を参照してください。

W&B Launchは三つのビルダーオプションをサポートしています：

* Docker: DockerビルダーはローカルのDockerデーモンを使用してイメージを構築します。
* [Kaniko](https://github.com/GoogleContainerTools/kaniko): KanikoはGoogleのプロジェクトで、Dockerデーモンが利用できない環境でのイメージ構築を可能にします。
* Noop: エージェントはジョブを構築せず、代わりに事前構築されたイメージのみをプルします。

{{% alert %}}
エージェントがDockerデーモンが利用できない環境（例えば、Kubernetesクラスター）でポーリングしている場合は、Kanikoビルダーを使用してください。

Kanikoビルダーの詳細は[Set up Kubernetes](./setup-launch-kubernetes.md)を参照してください。
{{% /alert %}}

イメージビルダーを指定するには、エージェントの設定にビルダーキーを含めます。例えば、以下のコードスニペットは、DockerまたはKanikoを使用することを指定しているLaunch設定 (`launch-config.yaml`) の一部を示しています：

```yaml title="launch-config.yaml"
builder:
  type: docker | kaniko | noop
```

### コンテナーレジストリを設定する
場合によっては、Launchエージェントをクラウドレジストリに接続したいことがあります。共通のシナリオには以下が含まれます：

* 構築した環境とは異なる環境（例えば、高性能なワークステーションやクラスター）でジョブを実行したい。
* エージェントを使用してイメージを構築し、Amazon SageMakerやVertexAIでこれらのイメージを実行したい。
* イメージリポジトリからプルするためのクレデンシャルをLaunchエージェントが提供するようにしたい。

コンテナーレジストリとやり取りするためにエージェントを設定する方法についての詳細は、[Advanced agent set](./setup-agent-advanced.md)をご覧ください。

## Launchエージェントを起動する
`launch-agent` W&B CLIコマンドを使用して、Launchエージェントを起動します：

```bash
wandb launch-agent -q <queue-1> -q <queue-2> --max-jobs 5
```

いくつかのユースケースでは、Kubernetesクラスター内からキューをポーリングするLaunchエージェントを持ちたい場合があります。詳細については[Advanced queue set up page](./setup-queue-advanced.md)を参照してください。