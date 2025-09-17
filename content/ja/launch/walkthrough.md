---
title: 'チュートリアル: W&B Launch の基礎'
description: W&B Launch の入門ガイド。
menu:
  launch:
    identifier: ja-launch-walkthrough
    parent: launch
url: guides/launch/walkthrough
weight: 1
---

## Launch とは？

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

W&B Launch を使うと、デスクトップでのトレーニング [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を、Amazon SageMaker や Kubernetes などのコンピュートリソースへ簡単にスケールできます。W&B Launch を設定すると、数回のクリックとコマンドで、トレーニングスクリプトの実行、モデルの評価スイートの実行、プロダクション推論向けのモデル準備などが素早く行えます。

## 仕組み

Launch は 3 つの基本コンポーネント、**launch ジョブ**、**キュー**、**エージェント** から構成されています。

[*launch job*]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) は、機械学習ワークフローでタスクを設定し実行するための設計図です。launch ジョブを作成したら、それを [*launch queue*]({{< relref path="./launch-terminology.md#launch-queue" lang="ja" >}}) に追加できます。launch キューは先入れ先出し（FIFO）のキューで、Amazon SageMaker や Kubernetes クラスターなどの特定のコンピュートターゲットリソースに対して、ジョブの設定と送信を行えます。

キューにジョブが追加されると、[*launch agents*]({{< relref path="./launch-terminology.md#launch-agent" lang="ja" >}}) がそのキューをポーリングし、キューで指定されたターゲットシステム上でジョブを実行します。

{{< img src="/images/launch/launch_overview.png" alt="W&B Launch の概要図" >}}

ユースケースに応じて、選択した [compute resource target]({{< relref path="./launch-terminology.md#target-resources" lang="ja" >}})（例: Amazon SageMaker）に合わせて launch キューを設定し、自身のインフラストラクチャーに launch エージェントをデプロイします。

Launch についての詳細は、[Terms and concepts]({{< relref path="./launch-terminology.md" lang="ja" >}}) を参照してください。

## 開始方法

ユースケースに応じて、以下のリソースから W&B Launch の利用を始めてください。

* 初めて W&B Launch を使う場合は、[Launch walkthrough]({{< relref path="#walkthrough" lang="ja" >}}) ガイドをおすすめします。
* [W&B Launch]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) のセットアップ方法を学びます。
* [launch job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) を作成します。
* W&B Launch の [public jobs GitHub repository](https://github.com/wandb/launch-jobs) で、[deploying to Triton](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton) や [evaluating an LLM](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) など、一般的なタスクのテンプレートをチェックしてください。
    * このリポジトリーから作成された launch ジョブは、公開 [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) で閲覧できます。

## Walkthrough

このページでは、W&B Launch のワークフローの基本を順に説明します。

{{% alert %}}
W&B Launch はコンテナー内で機械学習のワークロードを実行します。コンテナーの知識は必須ではありませんが、このウォークスルーでは役立つかもしれません。コンテナーの入門は [Docker のドキュメント](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/) を参照してください。
{{% /alert %}}

## 前提条件

始める前に、次の前提条件を満たしていることを確認してください。

1. https://wandb.ai/site でアカウントにサインアップし、W&B アカウントにログインします。
2. このウォークスルーには、Docker CLI とエンジンが動作するマシンへのターミナル アクセスが必要です。詳細は [Docker のインストールガイド](https://docs.docker.com/engine/install/) を参照してください。
3. W&B Python SDK の `0.17.1` 以上をインストールします:
```bash
pip install wandb>=0.17.1
```
4. ターミナルで `wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定して W&B に認証します。

{{< tabpane text=true >}}
{{% tab "Log in to W&B" %}}
    ターミナルで次を実行します:
    
    ```bash
    wandb login
    ```
{{% /tab %}}
{{% tab "Environment variable" %}}

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    `<your-api-key>` をあなたの W&B API key に置き換えてください。
{{% /tab %}}
{{% /tabpane %}}

## Create a launch job
[launch job]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) は次の 3 通りで作成できます: Docker イメージを使う、git リポジトリーから、ローカルのソースコードから。

{{< tabpane text=true >}}
{{% tab "With a Docker image" %}}
W&B にメッセージをログする事前作成済みコンテナーを実行するには、ターミナルを開いて次のコマンドを実行します:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

このコマンドはコンテナーイメージ `wandb/job_hello_world:main` をダウンロードして実行します。

Launch はコンテナーを設定し、`wandb` でログされたすべてを `launch-quickstart` project に送信します。コンテナーは W&B にメッセージをログし、W&B で新しく作成された run へのリンクを表示します。リンクをクリックして、W&B の UI で run を確認してください。
{{% /tab %}}
{{% tab "From a git repository" %}}
同じ hello-world のジョブを、[W&B Launch jobs リポジトリーのソースコード](https://github.com/wandb/launch-jobs) から起動するには、次のコマンドを実行します:

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
このコマンドは次を行います:
1. [W&B Launch jobs リポジトリー](https://github.com/wandb/launch-jobs) を一時ディレクトリーにクローンします。
2. **hello** project に **hello-world-git** という名前のジョブを作成します。このジョブは、コードの実行に使われた正確なソースコードと設定を追跡します。
3. `jobs/hello_world` ディレクトリーと `Dockerfile.wandb` からコンテナーイメージをビルドします。
4. コンテナーを起動し、`job.py` の Python スクリプトを実行します。

コンソール出力にはイメージのビルドと実行の様子が表示されます。コンテナーの出力は、前の例とほぼ同じになるはずです。

{{% /tab %}}
{{% tab "From local source code" %}}

git リポジトリーでバージョン管理されていないコードは、`--uri` 引数にローカルディレクトリーのパスを指定して起動できます。

空のディレクトリーを作成し、`train.py` という名前の Python スクリプトを次の内容で追加します:

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

さらに、次の内容で `requirements.txt` を追加します:

```text
wandb>=0.17.1
```

そのディレクトリー内で次のコマンドを実行します:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

このコマンドは次を行います:
1. カレントディレクトリーの内容を W&B に Code Artifact としてログします。
2. **launch-quickstart** project に **hello-world-code** という名前のジョブを作成します。
3. `train.py` と `requirements.txt` をベースイメージにコピーし、`pip install` で依存関係をインストールしてコンテナーイメージをビルドします。
4. コンテナーを起動し、`python train.py` を実行します。
{{% /tab %}}
{{< /tabpane >}}

## Create a queue

Launch は、共有コンピュート環境でチームがワークフローを構築できるように設計されています。これまでの例では、`wandb launch` コマンドがローカルマシン上でコンテナーを同期的に実行していました。Launch のキューとエージェントを使うと、共有リソース上での非同期実行や、優先度設定やハイパーパラメーター最適化といった高度な機能が利用できます。基本的なキューを作成するには、次の手順に従ってください。

1. [wandb.ai/launch](https://wandb.ai/launch) に移動し、**Create a queue** ボタンをクリックします。
2. キューを関連付ける **Entity** を選択します。
3. **Queue name** を入力します。
4. **Resource** として **Docker** を選択します。
5. **Configuration** は今は空のままにします。
6. **Create queue** をクリックします :rocket:

ボタンをクリックすると、ブラウザーはキュービューの **Agents** タブにリダイレクトします。エージェントがポーリングを開始するまで、キューは **Not active** の状態のままです。

{{< img src="/images/launch/create_docker_queue.gif" alt="Docker キューの作成" >}}

高度なキュー設定オプションについては、[advanced queue setup page]({{< relref path="/launch/set-up-launch/setup-queue-advanced.md" lang="ja" >}}) を参照してください。

## Connect an agent to the queue

キュービューでは、ポーリング中のエージェントがない場合、画面上部の赤いバナーに **Add an agent** ボタンが表示されます。ボタンをクリックすると、エージェントを実行するためのコマンドを表示してコピーできます。コマンドは次のようになります:

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

ターミナルでこのコマンドを実行してエージェントを起動します。エージェントは、実行すべきジョブがないか指定したキューをポーリングします。ジョブを受け取ると、エージェントはコンテナーイメージをダウンロードまたはビルドし、ローカルで `wandb launch` コマンドを実行したのと同様に、そのジョブを実行します。

[Launch ページ](https://wandb.ai/launch) に戻り、キューが **Active** と表示されていることを確認してください。

## Submit a job to the queue

W&B アカウントの新しい **launch-quickstart** project に移動し、画面左側のナビゲーションから Jobs タブを開きます。

**Jobs** ページには、これまでに実行した run から作成された W&B Jobs の一覧が表示されます。launch ジョブをクリックすると、ソースコード、依存関係、およびそのジョブから作成された run を確認できます。このウォークスルーを終えると、一覧には 3 つのジョブが表示されているはずです。

新しいジョブのいずれかを選び、次の手順でキューに送信します。

1. **Launch** ボタンをクリックしてジョブをキューに送信します。**Launch** ドロワーが表示されます。
2. 先ほど作成した **Queue** を選択し、**Launch** をクリックします。

これでジョブがキューに送信されます。このキューをポーリングしているエージェントがジョブを受け取り、実行します。ジョブの進捗は W&B の UI から、またはターミナルでエージェントの出力を確認することで追跡できます。

`wandb launch` コマンドは、`--queue` 引数を指定することで、ジョブを直接キューにプッシュできます。たとえば、hello-world のコンテナージョブをキューに送信するには、次のコマンドを実行します:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```