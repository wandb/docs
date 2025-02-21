---
title: 'Tutorial: W&B Launch basics'
description: W&B ローンンチのための入門ガイド。
menu:
  launch:
    identifier: ja-launch-walkthrough
    parent: launch
url: guides/launch/walkthrough
weight: 1
---

## Launch とは何ですか？

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

W&B Launch を使用して、デスクトップから Amazon SageMaker や Kubernetes などのコンピュートリソースにトレーニング [Runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を簡単にスケールできます。W&B Launch が設定されると、トレーニングスクリプト、モデル評価スイートを迅速に実行し、プロダクション推論用にモデルを準備することができ、数回のクリックとコマンドで操作可能です。

## 仕組み

Launch は、**launch jobs**、**queues**、および **agents** という 3 つの基本構成要素から成り立っています。

[*launch job*]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) は、機械学習ワークフローのタスクを設定して実行するための設計図です。Launch job を作成したら、[*launch queue*]({{< relref path="./launch-terminology.md#launch-queue" lang="ja" >}}) に追加することができます。Launch queue は、特定のコンピュートターゲットリソース（例: Amazon SageMaker や Kubernetes クラスター）にジョブを設定して送信するための先入れ先出し（FIFO）キューです。

ジョブがキューに追加されると、[*launch agents*]({{< relref path="./launch-terminology.md#launch-agent" lang="ja" >}}) がそのキューをポーリングし、キューでターゲットとされるシステム上でジョブを実行します。

{{< img src="/images/launch/launch_overview.png" alt="" >}}

ユースケースに基づいて、あなた（またはチームの誰か）が選んだ[コンピュートリソースターゲット]({{< relref path="./launch-terminology.md#target-resources" lang="ja" >}})（例: Amazon SageMaker）に従って launch queue を設定し、自分のインフラストラクチャーに launch agent をデプロイします。

Launch jobs やキューの動作方法、launch agent、および W&B Launch の動作についての詳細情報は、[用語とコンセプト]({{< relref path="./launch-terminology.md" lang="ja" >}}) ページをご覧ください。

## 開始方法

ユースケースに応じて、W&B Launch を始めるための以下のリソースを探索してください:

* W&B Launch を初めて使用する場合は、[Walkthrough]({{< relref path="#walkthrough" lang="ja" >}}) ガイドをご覧になることをお勧めします。
* [W&B Launch]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) の設定方法を学びます。
* [launch job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) を作成します。
* Triton へのデプロイや LLM の評価など一般的なタスクのテンプレートについては、W&B Launch の[公開 GitHub リポジトリ](https://github.com/wandb/launch-jobs) をご覧ください。
    * このリポジトリで作成された launch jobs を、この公開された[`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B project で確認します。

## Walkthrough

このページでは、W&B Launch ワークフローの基本を説明します。

{{% alert %}}
W&B Launch は、コンテナ内で機械学習ワークロードを実行します。コンテナに関する知識は必須ではありませんが、このウォークスルーで役立つかもしれません。コンテナの概要については、[Docker ドキュメント](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/) をご参照ください。
{{% /alert %}}

## 前提条件

始める前に、以下の前提条件を満たしていることを確認してください:

1. https://wandb.ai/site でアカウントを作成し、W&B アカウントにログインします。
2. このウォークスルーには、作業可能な Docker CLI とエンジンを備えたマシンへのターミナルアクセスが必要です。詳細については、[Docker インストールガイド](https://docs.docker.com/engine/install/) をご覧ください。
3. W&B Python SDK のバージョン `0.17.1` 以上をインストールします:
```bash
pip install wandb>=0.17.1
```
4. ターミナル内で `wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定して W&B に認証します。

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

    `<your-api-key>` をあなたの W&B APIキーに置き換えます。
{{% /tab %}}
{{% /tabpane %}}

## launch job を作成

Docker イメージを使用するか、git リポジトリから、またはローカルのソースコードから、3 つの方法のいずれかで[launch job]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) を作成します:

{{< tabpane text=true >}}
{{% tab "Docker イメージを使用" %}}
W&B にメッセージをログするプリメイクされたコンテナを実行するには、ターミナルを開いて次のコマンドを実行します:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

上記のコマンドは、コンテナイメージ `wandb/job_hello_world:main` をダウンロードして実行します。

Launch は、`wandb` でログされたすべてのものを `launch-quickstart` Project に報告するようにコンテナを設定します。コンテナは W&B にメッセージをログし、W&B で新たに作成された run へのリンクを表示します。リンクをクリックして W&B UI で run を表示します。
{{% /tab %}}
{{% tab "Git リポジトリから" %}}
W&B Launch jobs repository にあるソースコードから同じ hello-world job を起動するには、次のコマンドを実行します:

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
このコマンドは以下を行います:
1. [W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs) を一時ディレクトリーにクローンします。
2. **hello** Project の中に **hello-world-git** という名前のジョブを作成します。このジョブは、コードの実行に使用される正確なソースコードと設定を追跡します。
3. `jobs/hello_world` ディレクトリと `Dockerfile.wandb` からコンテナイメージをビルドします。
4. コンテナを開始し、`job.py` Pythonスクリプトを実行します。

コンソールの出力により、イメージのビルドと実行が表示されます。コンテナの出力は前の例とほぼ同一のはずです。

{{% /tab %}}
{{% tab "ローカルソースコードから" %}}

Git リポジトリでバージョン管理されていないコードは、`--uri` 引数にローカルディレクトリパスを指定することで起動できます。

空のディレクトリを作成し、`train.py` という名前の Python スクリプトを次の内容で追加します:

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

次の内容で `requirements.txt` ファイルを追加します:

```text
wandb>=0.17.1
```

ディレクトリ内から次のコマンドを実行します:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

このコマンドは以下を行います:
1. カレントディレクトリの内容を W&B に Code アーティファクトとしてログします。
2. **launch-quickstart** Project に **hello-world-code** という名前のジョブを作成します。
3. `train.py` と `requirements.txt` をベースイメージにコピーし、`pip install` で要件をインストールしてコンテナイメージをビルドします。
4. コンテナを開始し、`python train.py` を実行します。
{{% /tab %}}
{{< /tabpane >}}

## Queue を作成

Launch は、共有コンピュートを中心にワークフローを構築するのを支援するよう設計されています。これまでの例では、`wandb launch` コマンドがローカルマシンで同期的にコンテナを実行していました。Launch queues および agents により、ジョブを共有リソース上で非同期的に実行することができ、優先順位付けやハイパーパラメーター最適化などの高度な機能も利用できます。基本的な queue を作成するために、以下の手順に従ってください:

1. [wandb.ai/launch](https://wandb.ai/launch) に移動し、**Queue の作成** ボタンをクリックします。
2. Queue と関連付ける **Entity** を選択します。
3. **Queue 名** を入力します。
4. **Resource** として **Docker** を選択します。
5. **Configuration** は今のところ空白のままにします。
6. **Queue を作成** ボタンをクリックします :rocket:

ボタンをクリックすると、ブラウザーは queue ビューの **Agents** タブにリダイレクトされます。Queue は agent がポーリングを開始するまで **Not active** 状態に留まります。

{{< img src="/images/launch/create_docker_queue.gif" alt="" >}}

詳細な queue 設定オプションについては、[高度な queue 設定ページ]({{< relref path="./set-up-launch/setup-queue-advanced.md" lang="ja" >}}) を参照してください。

## エージェントを queue に接続

Queue ビューの上部にある赤いバナーに、polling agents がない場合は **Add an agent** ボタンが表示されます。ボタンをクリックして、エージェントを実行するためのコマンドをコピーして表示します。コマンドは次のように見えるはずです:

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

ターミナルでコマンドを実行してエージェントを開始します。エージェントは指定された queue をポーリングして実行するジョブを探します。受信したら、エージェントはジョブのためにコンテナイメージをダウンロードまたはビルドし、次に実行します。これはローカルで `wandb launch` コマンドを実行したかのようです。

[Launch ページ](https://wandb.ai/launch) に戻り、queue が **Active** と表示されていることを確認します。

## Queue にジョブを送信

W&B アカウントの新しい **launch-quickstart** Project に移動し、画面左側のナビゲーションからジョブタブを開きます。

**Jobs** ページには、以前に実行された run から作成された W&B Jobs のリストが表示されます。Launch job をクリックして、ソースコード、依存関係、およびジョブから作成された run を表示します。このウォークスルーを終了すると、リストには 3 つのジョブがあるはずです。

新しいジョブの 1 つを選び、次の手順で queue に送信します:

1. ジョブを queue に送信するために **Launch** ボタンをクリックします。**Launch** ドロワーが表示されます。
2. 先ほど作成した **Queue** を選択し、**Launch** をクリックします。

これにより、ジョブが queue に送信されます。Queue をポーリングしているエージェントがジョブを受け取り、実行します。ジョブの進行状況は W&B UI から、またはエージェントの出力をターミナルで確認することができます。

`wandb launch` コマンドは `--queue` 引数を指定することにより、直接ジョブを queue にプッシュできます。たとえば、hello-world コンテナジョブを queue に送信するには、次のコマンドを実行します:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```