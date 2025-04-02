---
title: 'Tutorial: W&B Launch basics'
description: W&B Launch のスタートアップ ガイド 。
menu:
  launch:
    identifier: ja-launch-walkthrough
    parent: launch
url: guides/launch/walkthrough
weight: 1
---

## Launch とは?

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

W&B Launch を使用すると、デスクトップから Amazon SageMaker や Kubernetes などのコンピューティングリソースまで、トレーニング [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) を簡単に拡張できます。 W&B Launch を設定すると、数回クリックしてコマンドを実行するだけで、トレーニング スクリプト、モデル 評価スイートの実行、本番環境での推論に向けたモデルの準備などをすばやく行うことができます。

## 仕組み

Launch は、**launch jobs** 、**queues** 、**agents** の 3 つの基本的なコンポーネントで構成されています。

[*launch job*]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) は、 ML ワークフローでタスクを構成および実行するための設計図です。 launch job を作成したら、それを [*launch queue*]({{< relref path="./launch-terminology.md#launch-queue" lang="ja" >}}) に追加できます。 launch queue は、先入れ先出し (FIFO) キューであり、Amazon SageMaker や Kubernetes クラスターなどの特定のコンピューティング ターゲット リソースにジョブを構成して送信できます。

ジョブが queue に追加されると、[*launch agents*]({{< relref path="./launch-terminology.md#launch-agent" lang="ja" >}}) はその queue をポーリングし、 queue をターゲットとするシステムでジョブを実行します。

{{< img src="/images/launch/launch_overview.png" alt="" >}}

ユースケースに基づいて、ユーザー (またはチームの誰か) が、選択した [コンピューティング リソース ターゲット]({{< relref path="./launch-terminology.md#target-resources" lang="ja" >}}) (たとえば、Amazon SageMaker) に従って launch queue を構成し、独自のインフラストラクチャに launch agent をデプロイします。

launch jobs 、 queue の仕組み、 launch agents 、および W&B Launch の仕組みに関する追加情報については、[用語と概念]({{< relref path="./launch-terminology.md" lang="ja" >}}) ページを参照してください。

## 開始方法

ユースケースに応じて、次のリソースを参照して W&B Launch を開始してください。

* W&B Launch を初めて使用する場合は、[チュートリアル]({{< relref path="#walkthrough" lang="ja" >}}) ガイドを参照することをお勧めします。
* [W&B Launch]({{< relref path="/launch/set-up-launch/" lang="ja" >}}) の設定方法について説明します。
* [launch job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) を作成します。
* W&B Launch の[パブリック jobs GitHub リポジトリ](https://github.com/wandb/launch-jobs)で、[Triton へのデプロイ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton)、[LLM の評価](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) などの一般的なタスクのテンプレートを確認してください。
    * このリポジトリから作成された launch jobs は、このパブリック [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B プロジェクトで表示します。

## チュートリアル

このページでは、W&B Launch ワークフローの基本について説明します。

{{% alert %}}
W&B Launch は、機械学習のワークロードをコンテナで実行します。 コンテナに精通している必要はありませんが、このチュートリアルに役立つ場合があります。 コンテナの入門書については、[Docker ドキュメント](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/)を参照してください。
{{% /alert %}}

## 前提条件

開始する前に、次の前提条件を満たしていることを確認してください。

1. https://wandb.ai/site でアカウントにサインアップし、W&B アカウントにログインします。
2. このチュートリアルでは、動作する Docker CLI およびエンジンを備えたマシンへの ターミナル  アクセスが必要です。 詳しくは、[Docker インストール ガイド](https://docs.docker.com/engine/install/)をご覧ください。
3. W&B Python SDK バージョン `0.17.1` 以上をインストールします。
```bash
pip install wandb>=0.17.1
```
4. ターミナル  内で `wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定して W&B で認証します。

{{< tabpane text=true >}}
{{% tab "W&B へのログイン" %}}
    ターミナル  内で以下を実行します。
    
    ```bash
    wandb login
    ```
{{% /tab %}}
{{% tab "環境変数" %}}

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    `<your-api-key>` を W&B APIキーに置き換えます。
{{% /tab %}}
{{< /tabpane >}}

## launch job の作成
次の 3 つの方法のいずれかで [launch job]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) を作成します。Docker イメージを使用、git リポジトリから、またはローカル ソース コードから。

{{< tabpane text=true >}}
{{% tab "Docker イメージを使用" %}}
W&B にメッセージを記録する既製のコンテナを実行するには、 ターミナル  を開き、次のコマンドを実行します。

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

上記のコマンドは、コンテナ イメージ `wandb/job_hello_world:main` をダウンロードして実行します。

Launch は、`wandb` で記録されたすべての内容を `launch-quickstart` project に報告するようにコンテナを構成します。 コンテナはメッセージを W&B に記録し、新しく作成された run へのリンクを W&B に表示します。 リンクをクリックして、W&B UI で run を表示します。
{{% /tab %}}
{{% tab "git リポジトリから" %}}
[W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs) のソース コードから同じ hello-world ジョブを起動するには、次のコマンドを実行します。

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
このコマンドは次のことを行います。
1. [W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs)を一時ディレクトリーにクローンします。
2. **hello** project に **hello-world-git** という名前のジョブを作成します。 このジョブは、コードの実行に使用される正確なソース コードと設定を追跡します。
3. `jobs/hello_world` ディレクトリーと `Dockerfile.wandb` からコンテナ イメージを構築します。
4. コンテナを起動し、`job.py` Python スクリプトを実行します。

コンソール出力に、イメージの構築と実行が表示されます。 コンテナの出力は、前の例とほぼ同じであるはずです。

{{% /tab %}}
{{% tab "ローカル ソース コードから" %}}

git リポジトリでバージョン管理されていないコードは、`--uri` 引数にローカル ディレクトリー パスを指定することで起動できます。

空のディレクトリーを作成し、次の内容で `train.py` という名前の Python スクリプトを追加します。

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

次の内容で `requirements.txt` ファイルを追加します。

```text
wandb>=0.17.1
```

ディレクトリー内から、次のコマンドを実行します。

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

このコマンドは次のことを行います。
1. 現在のディレクトリーの内容をコード Artifacts として W&B に記録します。
2. **launch-quickstart** project に **hello-world-code** という名前のジョブを作成します。
3. `train.py` と `requirements.txt` をベース イメージにコピーし、要件を `pip install` してコンテナ イメージを構築します。
4. コンテナを起動し、`python train.py` を実行します。
{{% /tab %}}
{{< /tabpane %}}

## queue の作成

Launch は、チームが共有コンピューティングを中心にワークフローを構築するのに役立つように設計されています。 これまでの例では、`wandb launch` コマンドはローカル マシンでコンテナを同期的に実行していました。 Launch queues と agents により、共有リソースでのジョブの非同期実行と、優先順位付けやハイパーパラメーター最適化などの高度な機能が実現します。 基本的な queue を作成するには、次の手順に従います。

1. [wandb.ai/launch](https://wandb.ai/launch) に移動し、**queue の作成**ボタンをクリックします。
2. queue に関連付ける **Entity** を選択します。
3. **queue 名**を入力します。
4. **リソース**として **Docker** を選択します。
5. **構成**は、今のところ空白のままにします。
6. **queue の作成** :rocket: をクリックします

ボタンをクリックすると、ブラウザは queue ビューの **Agents** タブにリダイレクトされます。 agent がポーリングを開始するまで、 queue は **非アクティブ** 状態のままです。

{{< img src="/images/launch/create_docker_queue.gif" alt="" >}}

高度な queue 構成オプションについては、[高度な queue セットアップ ページ]({{< relref path="/launch/set-up-launch/setup-queue-advanced.md" lang="ja" >}}) を参照してください。

## agent を queue に接続する

queue にポーリング agent がない場合、 queue ビューの画面上部の赤いバナーに **agent の追加**ボタンが表示されます。 ボタンをクリックして、agent を実行するコマンドをコピーして表示します。 コマンドは次のようになります。

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

ターミナル  でコマンドを実行して、agent を起動します。 agent は、実行するジョブについて、指定された queue をポーリングします。 受信すると、agent はジョブのコンテナ イメージをダウンロードまたは構築してから実行します。これは、`wandb launch` コマンドがローカルで実行された場合と同様です。

[Launch ページ](https://wandb.ai/launch) に戻り、 queue が **アクティブ** と表示されることを確認します。

## ジョブを queue に送信する

W&B アカウントで新しい **launch-quickstart** project に移動し、画面左側のナビゲーションから [Jobs] タブを開きます。

[Jobs] ページには、以前に実行された runs から作成された W&B Jobs のリストが表示されます。 launch job をクリックして、ソース コード、依存関係、およびジョブから作成された runs を表示します。 このチュートリアルを完了すると、リストに 3 つのジョブが表示されます。

新しいジョブの 1 つを選択し、次の手順に従って queue に送信します。

1. [**Launch**] ボタンをクリックして、ジョブを queue に送信します。 **Launch** ドロワーが表示されます。
2. 以前に作成した **queue** を選択し、[**Launch**] をクリックします。

これにより、ジョブが queue に送信されます。 この queue をポーリングする agent は、ジョブを取得して実行します。 ジョブの進捗状況は、W&B UI から監視するか、 ターミナル  で agent の出力を検査することで監視できます。

`wandb launch` コマンドは、`--queue` 引数を指定することで、ジョブを queue に直接プッシュできます。 たとえば、hello-world コンテナ ジョブを queue に送信するには、次のコマンドを実行します。

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```
