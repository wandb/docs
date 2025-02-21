---
title: 'Tutorial: W&B Launch basics'
description: W&B の Launch の入門ガイド。
menu:
  launch:
    identifier: ja-launch-walkthrough
    parent: launch
url: guides/launch/walkthrough
weight: 1
---

## Launch とは

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

W&B Launch を使用すると、デスクトップから Amazon SageMaker や Kubernetes などのコンピューティングリソースに、[run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) のトレーニングを簡単に拡張できます。 W&B Launch を構成すると、数回クリックするだけで、トレーニングスクリプト、モデルの評価スイートの実行、本番環境の推論のためのモデルの準備などを迅速に行うことができます。

## 仕組み

Launch は、**launch jobs** 、**queues** 、**agents** の3つの基本的なコンポーネントで構成されています。

[*launch job*]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) は、 ML ワークフローでタスクを構成および実行するための設計図です。 launch job を作成したら、それを[*launch queue*]({{< relref path="./launch-terminology.md#launch-queue" lang="ja" >}}) に追加できます。 launch queue は、Amazon SageMaker や Kubernetes クラスターなどの特定のコンピューティングターゲットリソースに対してジョブを構成および送信できる、先入れ先出し (FIFO) キューです。

ジョブがキューに追加されると、[*launch agent*]({{< relref path="./launch-terminology.md#launch-agent" lang="ja" >}}) がそのキューをポーリングし、キューがターゲットとするシステムでジョブを実行します。

{{< img src="/images/launch/launch_overview.png" alt="" >}}

ユースケースに応じて、あなた (またはあなたのチームの誰か) が、選択した[コンピューティングリソースターゲット]({{< relref path="./launch-terminology.md#target-resources" lang="ja" >}}) (たとえば、Amazon SageMaker) に従って launch queue を構成し、独自のインフラストラクチャに launch agent をデプロイします。

launch jobs 、キューの仕組み、 launch agent 、および W&B Launch の仕組みに関する追加情報については、[用語と概念]({{< relref path="./launch-terminology.md" lang="ja" >}}) ページを参照してください。

## 開始方法

ユースケースに応じて、次のリソースを参照して W&B Launch を開始してください。

* W&B Launch を初めて使用する場合は、[チュートリアル]({{< relref path="#walkthrough" lang="ja" >}}) ガイドを参照することをお勧めします。
* [W&B Launch]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) のセットアップ方法を学びます。
* [launch job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) を作成します。
* W&B Launch [パブリック jobs GitHub リポジトリ](https://github.com/wandb/launch-jobs) で、[Triton へのデプロイ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton) や [LLM の評価](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) などの一般的なタスクのテンプレートを確認してください。
    * このリポジトリから作成された launch jobs は、このパブリック [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B プロジェクトで表示できます。

## チュートリアル

このページでは、W&B Launch ワークフローの基本を説明します。

{{% alert %}}
W&B Launch は、コンテナ内で機械学習ワークロードを実行します。 コンテナに精通している必要はありませんが、このチュートリアルに役立つ場合があります。 コンテナの入門書については、[Docker ドキュメント](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/) を参照してください。
{{% /alert %}}

## 前提条件

開始する前に、次の前提条件を満たしていることを確認してください。

1. https://wandb.ai/site でアカウントにサインアップし、W&B アカウントにログインします。
2. このチュートリアルでは、動作する Docker CLI およびエンジンを備えたマシンへの ターミナル アクセスが必要です。 詳しくは、[Docker インストールガイド](https://docs.docker.com/engine/install/) をご覧ください。
3. W&B Python SDK バージョン `0.17.1` 以上をインストールします。
```bash
pip install wandb>=0.17.1
```
4. ターミナル内で `wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定して W&B で認証します。

{{< tabpane text=true >}}
{{% tab "W&B へのログイン" %}}
    ターミナル内で次を実行します。
    
    ```bash
    wandb login
    ```
{{% /tab %}}
{{% tab "環境変数" %}}

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    `<your-api-key>` を W&B API キーに置き換えます。
{{% /tab %}}
{{% /tabpane %}}

## launch job を作成する
Docker イメージ、git リポジトリ、またはローカルソースコードを使用して、次の3つの方法のいずれかで [launch job]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) を作成します。

{{< tabpane text=true >}}
{{% tab "Docker イメージを使用する" %}}
W&B にメッセージを ログ する既製のコンテナを実行するには、 ターミナル を開き、次のコマンドを実行します。

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

上記のコマンドは、コンテナイメージ `wandb/job_hello_world:main` をダウンロードして実行します。

Launch は、`wandb` で ログ されたすべてを `launch-quickstart` project に報告するようにコンテナを構成します。 コンテナは W&B にメッセージを ログ し、新しく作成された run へのリンクを W&B に表示します。 リンクをクリックして、W&B UI で run を表示します。
{{% /tab %}}
{{% tab "git リポジトリから" %}}
[W&B Launch jobs リポジトリのソースコード](https://github.com/wandb/launch-jobs) から同じ hello-world job を起動するには、次のコマンドを実行します。

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
このコマンドは次のことを行います。
1. [W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs) を一時ディレクトリーにクローンします。
2. **hello** project に **hello-world-git** という名前の job を作成します。 この job は、コードの実行に使用される正確なソースコードと構成を追跡します。
3. `jobs/hello_world` ディレクトリーと `Dockerfile.wandb` からコンテナイメージを構築します。
4. コンテナを起動し、`job.py` python スクリプトを実行します。

コンソール出力には、イメージのビルドと実行が表示されます。 コンテナの出力は、前の例とほぼ同じである必要があります。

{{% /tab %}}
{{% tab "ローカルソースコードから" %}}

git リポジトリでバージョン管理されていないコードは、`--uri` 引数にローカルディレクトリーパスを指定して起動できます。

空のディレクトリーを作成し、次の内容の `train.py` という名前の Python スクリプトを追加します。

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

次の内容の `requirements.txt` ファイルを追加します。

```text
wandb>=0.17.1
```

ディレクトリー内から、次のコマンドを実行します。

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

このコマンドは次のことを行います。
1. 現在のディレクトリーの内容を Code Artifact として W&B に ログ します。
2. **launch-quickstart** project に **hello-world-code** という名前の job を作成します。
3. `train.py` と `requirements.txt` をベースイメージにコピーし、要件を `pip install` して、コンテナイメージを構築します。
4. コンテナを起動し、`python train.py` を実行します。
{{% /tab %}}
{{< /tabpane >}}

## queue を作成する

Launch は、チームが共有コンピュートを中心に ワークフロー を構築するのに役立つように設計されています。 これまでの例では、`wandb launch` コマンドはローカルマシンでコンテナを同期的に実行していました。 Launch queues と agent を使用すると、共有リソースでのジョブの非同期実行や、優先順位付けや ハイパーパラメーター 最適化などの高度な機能が可能になります。 基本的な queue を作成するには、次の手順に従います。

1. [wandb.ai/launch](https://wandb.ai/launch) に移動し、**queue を作成** ボタンをクリックします。
2. queue に関連付ける **Entity** を選択します。
3. **queue 名** を入力します。
4. **リソース** として **Docker** を選択します。
5. 今のところ、**構成** は空白のままにします。
6. **queue を作成** :rocket: をクリックします

ボタンをクリックすると、ブラウザーは queue ビューの **Agents** タブにリダイレクトされます。 agent がポーリングを開始するまで、 queue は **非アクティブ** 状態のままです。

{{< img src="/images/launch/create_docker_queue.gif" alt="" >}}

高度な queue 構成オプションについては、[高度な queue のセットアップページ]({{< relref path="./set-up-launch/setup-queue-advanced.md" lang="ja" >}}) を参照してください。

## agent を queue に接続する

queue にポーリング agent がない場合、 queue ビューには画面上部の赤いバナーに **agent を追加** ボタンが表示されます。 ボタンをクリックして、agent を実行するコマンドをコピーして表示します。 コマンドは次のようになります。

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

ターミナル でコマンドを実行して、agent を起動します。 agent は、実行するジョブについて、指定された queue をポーリングします。 受信すると、agent は `wandb launch` コマンドがローカルで実行されたかのように、ジョブのコンテナイメージをダウンロードまたは構築して実行します。

[Launch ページ](https://wandb.ai/launch) に戻り、 queue が **アクティブ** と表示されていることを確認します。

## ジョブを queue に送信する

W&B アカウントで新しい **launch-quickstart** project に移動し、画面左側のナビゲーションから jobs タブを開きます。

**Jobs** ページには、以前に実行された run から作成された W&B Jobs のリストが表示されます。 launch job をクリックして、ソースコード、依存関係、および job から作成された run を表示します。 このチュートリアルを完了すると、リストに3つの jobs が表示されます。

新しい jobs のいずれかを選択し、次の手順に従って queue に送信します。

1. **Launch** ボタンをクリックして、ジョブを queue に送信します。 **Launch** ドロワーが表示されます。
2. 先ほど作成した **queue** を選択し、**Launch** をクリックします。

これにより、ジョブが queue に送信されます。 この queue をポーリングする agent は、ジョブを取得して実行します。 ジョブの進行状況は、W&B UI から監視するか、 ターミナル で agent の出力を調べることで監視できます。

`wandb launch` コマンドは、`--queue` 引数を指定することで、ジョブを queue に直接プッシュできます。 たとえば、hello-world コンテナ job を queue に送信するには、次のコマンドを実行します。

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```
