---
title: 'チュートリアル: W&B ローンンチ 基本事項'
description: W&B ローンンチの入門ガイド。
menu:
  launch:
    identifier: ja-launch-walkthrough
    parent: launch
url: guides/launch/walkthrough
weight: 1
---

## What is Launch?

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

W&B Launch を使用して、トレーニング [Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) をデスクトップから Amazon SageMaker、Kubernetes などの計算リソースに簡単にスケールできます。W&B Launch の設定が完了すると、トレーニング スクリプト、モデルの評価スイート、プロダクション推論用のモデルの準備などを、数回のクリックとコマンドで迅速に実行できます。

## 仕組み

Launch は、**launch jobs**、**queues**、**agents** の3つの基本的なコンポーネントで構成されています。

[*launch job*]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}})は、ML ワークフローでタスクを設定および実行するためのブループリントです。Launch Job を作成したら、[*launch queue*]({{< relref path="./launch-terminology.md#launch-queue" lang="ja" >}}) に追加できます。Launch Queue は、Amazon SageMaker や Kubernetes クラスターなどの特定のコンピュートターゲットリソースに Jobs を構成して送信できる先入れ先出し (FIFO) のキューです。

ジョブがキューに追加されると、[*launch agents*]({{< relref path="./launch-terminology.md#launch-agent" lang="ja" >}}) がそのキューをポーリングし、キューによってターゲットとされたシステムでジョブを実行します。

{{< img src="/images/launch/launch_overview.png" alt="" >}}

ユースケースに基づいて、あなた自身またはチームの誰かが選択した [compute resource target]({{< relref path="./launch-terminology.md#target-resources" lang="ja" >}})（たとえば、Amazon SageMaker）に従って Launch Queue を設定し、独自のインフラストラクチャーに Launch エージェントをデプロイします。

Launch Jobs、キューの仕組み、Launch エージェント、および W&B Launch の動作に関する追加情報については、[Terms and Concepts]({{< relref path="./launch-terminology.md" lang="ja" >}}) ページを参照してください。

## 開始方法

ユースケースに応じて、W&B Launch を始めるために次のリソースを確認してください。

* 初めて W&B Launch を使用する場合は、[Walkthrough]({{< relref path="#walkthrough" lang="ja" >}}) ガイドを閲覧することをお勧めします。
* [W&B Launch]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) の設定方法を学びます。
* [Launch Job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) を作成します。
* Triton へのデプロイや LLM の評価などの一般的なタスクのテンプレートについては、W&B Launch の[公開ジョブ GitHub リポジトリ](https://github.com/wandb/launch-jobs) をチェックしてください。
    * このリポジトリから作成された Launch Job は、この公開された [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B プロジェクトで閲覧できます。

## Walkthrough

このページでは、W&B Launch ワークフローの基本を案内します。

{{% alert %}}
W&B Launch は、コンテナ内で機械学習ワークロードを実行します。コンテナについての知識は必須ではありませんが、このWalkthroughに役立ちます。コンテナの入門書は [Docker ドキュメント](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/) をご覧ください。
{{% /alert %}}

## Prerequisites

開始する前に、次の前提条件が満たされていることを確認してください。

1. https://wandb.ai/site でアカウントに登録し、その後 W&B アカウントにログインします。
2. この Walkthrough には、動作する Docker CLI とエンジン付きのマシンへのターミナル アクセスが必要です。詳細については [Docker インストールガイド](https://docs.docker.com/engine/install/) を参照してください。
3. W&B Python SDK バージョン `0.17.1` 以上をインストールします:
```bash
pip install wandb>=0.17.1
```
4. ターミナル内で `wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定して W&B を認証します。

{{< tabpane text=true >}}
{{% tab "Log in to W&B" %}}
    ターミナル内で以下を実行します:

    ```bash
    wandb login
    ```
{{% /tab %}}
{{% tab "Environment variable" %}}

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    `<your-api-key>` をあなたの W&B API キーに置き換えます。
{{% /tab %}}
{{% /tabpane %}}

## Create a launch job
Docker イメージ、git リポジトリから、またはローカルソースコードから3つの方法のいずれかで [Launch Job]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) を作成します。

{{< tabpane text=true >}}
{{% tab "With a Docker image" %}}
W&B にメッセージをログする事前に作成されたコンテナを実行するには、ターミナルを開いて次のコマンドを実行します。

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

前述のコマンドは、コンテナイメージ `wandb/job_hello_world:main` をダウンロードして実行します。

Launch は、`wandb` でログされたすべての情報を `launch-quickstart` プロジェクトに報告するようにコンテナを設定します。コンテナは W&B にメッセージをログし、新しく作成された Run へのリンクを W&B に表示します。リンクをクリックして、W&B UI で Run を確認します。
{{% /tab %}}
{{% tab "From a git repository" %}}
同じ hello-world ジョブを [W&B Launch jobs リポジトリ内のソースコード](https://github.com/wandb/launch-jobs) から起動するには、次のコマンドを実行します。

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
このコマンドは次のことを行います。
1. [W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs) を一時ディレクトリにクローンします。
2. **hello** プロジェクト内に **hello-world-git** という名前のジョブを作成します。このジョブは、コードの実行に使用される正確なソースコードと設定を追跡します。
3. `jobs/hello_world` ディレクトリと `Dockerfile.wandb` からコンテナイメージをビルドします。
4. コンテナを開始し、`job.py` Python スクリプトを実行します。

コンソール出力には、イメージのビルドと実行が表示されます。コンテナの出力は、前の例とほぼ同じである必要があります。

{{% /tab %}}
{{% tab "From local source code" %}}

git リポジトリにバージョン管理されていないコードは、`--uri` 引数にローカルディレクトリパスを指定することで起動できます。

空のディレクトリを作成し、次の内容を持つ `train.py` という名前の Python スクリプトを追加します。

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

次の内容で `requirements.txt` ファイルを追加します。

```text
wandb>=0.17.1
```

ディレクトリ内から次のコマンドを実行します。

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

このコマンドは次のことを行います。
1. 現在のディレクトリの内容を W&B に Code Artifact としてログします。
2. **launch-quickstart** プロジェクトに **hello-world-code** という名前のジョブを作成します。
3. `train.py` と `requirements.txt` を基礎イメージにコピーしてコンテナイメージをビルドし、`pip install` で要件をインストールします。
4. コンテナを開始して `python train.py` を実行します。
{{% /tab %}}
{{< /tabpane >}}

## Create a queue

Launch は、Teams が共有計算を中心にワークフローを構築するのを支援するように設計されています。これまでの例では、`wandb launch` コマンドがローカルマシンでコンテナを同期的に実行しました。Launch キューとエージェントを使用すると、共有リソースでジョブを非同期に実行し、優先順位付けやハイパーパラメータ最適化などの高度な機能を実現できます。基本的なキューを作成するには、次の手順に従います。

1. [wandb.ai/launch](https://wandb.ai/launch) にアクセスし、**Create a queue** ボタンをクリックします。
2. キューを関連付ける **Entity** を選択します。
3. **Queue name** を入力します。
4. **Resource** として **Docker** を選択します。
5. 今のところ、**Configuration** は空白のままにします。
6. **Create queue** をクリックします :rocket:

ボタンをクリックすると、ブラウザはキュー表示の **Agents** タブにリダイレクトされます。キューにエージェントがポーリングされるまで、キューは **Not active** 状態のままです。

{{< img src="/images/launch/create_docker_queue.gif" alt="" >}}

高度なキューの設定オプションについては、[advanced queue setup ページ]({{< relref path="/launch/set-up-launch/setup-queue-advanced.md" lang="ja" >}}) を参照してください。

## Connect an agent to the queue

キューにポーリング エージェントがない場合、キュー ビューには画面上部の赤いバナーに **Add an agent** ボタンが表示されます。ボタンをクリックしてコマンドをコピーし、エージェントを実行します。コマンドは次のようになります。

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

コマンドをターミナルで実行してエージェントを起動します。エージェントは指定されたキューをポーリングして、実行するジョブを収集します。受信後、エージェントはジョブのためにコンテナイメージをダウンロードまたはビルドして実行します。`wandb launch` コマンドがローカルで実行されたかのように。

[Launch ページ](https://wandb.ai/launch) に戻って、キューが **Active** として表示されていることを確認します。

## Submit a job to the queue

W&B アカウントの **launch-quickstart** プロジェクトに移動し、画面の左側のナビゲーションから Jobs タブを開きます。

**Jobs** ページには、以前に実行された Runs から作成された W&B Jobs のリストが表示されます。Launch Job をクリックすると、ソースコード、依存関係、およびジョブから作成されたすべての Runs を表示できます。この Walkthrough を完了すると、リストに3つのジョブが表示されるはずです。

新しいジョブのいずれかを選択し、それをキューに送信する手順は次のとおりです。

1. **Launch** ボタンをクリックして、ジョブをキューに送信します。 **Launch** ドロワーが表示されます。
2. 先ほど作成した **Queue** を選択し、**Launch** をクリックします。

これにより、ジョブがキューに送信されます。このキューをポーリングするエージェントがジョブを取得し、実行します。ジョブの進行状況は、W&B UI からやターミナル内のエージェントの出力を調査することで監視できます。

`wandb launch` コマンドは `--queue` 引数を指定することで Jobs をキューに直接プッシュできます。たとえば、hello-world コンテナジョブをキューに送信するには、次のコマンドを実行します。

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```