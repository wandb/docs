---
title: 'チュートリアル: W&B Launch の基本'
description: W&B ローンンチの使い方ガイド。
menu:
  launch:
    identifier: ja-launch-walkthrough
    parent: launch
url: guides/launch/walkthrough
weight: 1
---

## Launch とは？

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

W&B Launch を使うことで、デスクトップから Amazon SageMaker や Kubernetes などの計算リソースに [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) のトレーニングを手軽にスケールできます。W&B Launch の設定が完了すれば、トレーニングスクリプトやモデル評価の実行、プロダクション推論用のモデル準備などを、数回のクリックやコマンドで素早く実施できます。

## 仕組み

Launch は、**launch jobs**、**queues**、**agents** という3つの基本コンポーネントから構成されています。

[*launch job*]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) は、機械学習ワークフローでタスクを設定し実行するための設計図です。launch job を作成したら、[*launch queue*]({{< relref path="./launch-terminology.md#launch-queue" lang="ja" >}}) に追加できます。launch queue は FIFO（先入れ先出し）キューで、特定の計算リソース（例: Amazon SageMaker、Kubernetes クラスターなど）をターゲットとして、ジョブの設定と送信ができます。

ジョブがキューに追加されると、[*launch agents*]({{< relref path="./launch-terminology.md#launch-agent" lang="ja" >}}) がそのキューを監視し、キューが指定するシステム上でジョブを実行します。

{{< img src="/images/launch/launch_overview.png" alt="W&B Launch overview diagram" >}}

ユースケースに応じて、あなた（またはチームの誰か）が選択した[計算リソースターゲット]({{< relref path="./launch-terminology.md#target-resources" lang="ja" >}})（例: Amazon SageMaker）に合わせて launch queue を設定し、自身のインフラ上に launch agent をデプロイします。

Launch についてさらに詳しくは、[用語とコンセプト]({{< relref path="./launch-terminology.md" lang="ja" >}})ページを参照してください。

## 開始方法

目的に応じて、以下のリソースから W&B Launch の使い始め方を確認できます：

* W&B Launch を初めて利用する場合は、[Launch walkthrough]({{< relref path="#walkthrough" lang="ja" >}}) ガイドを参照することをおすすめします。
* [W&B Launch のセットアップ]({{< relref path="/launch/set-up-launch/" lang="ja" >}})方法を学びましょう。
* [launch job]({{< relref path="./create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) を作成しましょう。
* W&B Launch の [public jobs GitHub リポジトリ](https://github.com/wandb/launch-jobs)で、[Triton へのデプロイ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton) や [LLM の評価](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) などの一般的なタスクのテンプレートをチェックしましょう。
    * このリポジトリから作成された launch job を [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) の公開 W&B プロジェクトで確認できます。

## Walkthrough

このページでは W&B Launch ワークフローの基本を説明します。

{{% alert %}}
W&B Launch は機械学習のワークロードをコンテナで実行します。コンテナに詳しくなくても問題ありませんが、理解しておくとよりスムーズです。基本的な内容は [Docker ドキュメント](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/)を参照してください。
{{% /alert %}}

## 前提条件

はじめる前に、以下の前提条件を満たしていることを確認してください：

1. https://wandb.ai/site でアカウント登録し、W&B アカウントにログインしてください。
2. この walkthrough には、Docker CLI およびエンジンが動作するマシンへのターミナルアクセスが必要です。詳しくは [Docker インストールガイド](https://docs.docker.com/engine/install/)を確認してください。
3. W&B Python SDK のバージョン `0.17.1` 以上をインストールしてください：
```bash
pip install wandb>=0.17.1
```
4. ターミナルで `wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定して W&B に認証してください。

{{< tabpane text=true >}}
{{% tab "W&B にログイン" %}}
    ターミナルで以下のコマンドを実行します：

    ```bash
    wandb login
    ```
{{% /tab %}}
{{% tab "環境変数" %}}

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    `<your-api-key>` を、ご自身の W&B APIキーに置き換えてください。
{{% /tab %}}
{{% /tabpane %}}

## launch job の作成

[launch job]({{< relref path="./launch-terminology.md#launch-job" lang="ja" >}}) は、Docker イメージ・git リポジトリ・ローカルソースコードのいずれかから作成できます。

{{< tabpane text=true >}}
{{% tab "Docker イメージから" %}}
事前に用意されたコンテナで W&B へのログを確認するには、ターミナルを開いて以下のコマンドを実行してください：

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

このコマンドは `wandb/job_hello_world:main` のコンテナイメージをダウンロード・実行します。

Launch は、このコンテナを設定して `wandb` でログされた内容を `launch-quickstart` プロジェクトに記録します。コンテナ内で W&B へのメッセージが記録され、新しい run へのリンクが表示されます。リンクをクリックすると W&B UI で run を確認できます。
{{% /tab %}}
{{% tab "git リポジトリから" %}}
同じ hello-world のジョブを [W&B Launch jobs リポジトリ内のソースコード](https://github.com/wandb/launch-jobs)から実行する場合、以下のコマンドを実行してください：

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
このコマンドの動作は以下の通りです：
1. [W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs)を一時ディレクトリーにクローン。
2. **hello-world-git** というジョブを **hello** プロジェクト内に作成。このジョブは実行で使われた正確なソースコードと設定を追跡します。
3. `jobs/hello_world` ディレクトリーと `Dockerfile.wandb` からコンテナイメージをビルド。
4. コンテナを起動し `job.py` Python スクリプトを実行。

コンソールにはイメージのビルド・実行状況が表示されます。出力内容は前の例とほぼ同じです。

{{% /tab %}}
{{% tab "ローカルソースコードから" %}}

git リポジトリでバージョン管理されていないコードでも、`--uri` 引数にローカルのディレクトリパスを指定すれば launch できます。

空のディレクトリを作成し、次の内容で `train.py` という Python スクリプトを作成してください：

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

`requirements.txt` というファイルに以下を追加します：

```text
wandb>=0.17.1
```

ディレクトリ内で、次のコマンドを実行してください：

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

このコマンドで実施される内容：
1. 現在のディレクトリの内容を W&B の Code Artifact としてログ。
2. **hello-world-code** というジョブを **launch-quickstart** プロジェクト内に作成。
3. `train.py` と `requirements.txt` をベースイメージにコピーし、`pip install` で依存関係をインストールしてコンテナイメージをビルド。
4. コンテナを起動し `python train.py` を実行。
{{% /tab %}}
{{< /tabpane >}}

## Queue の作成

Launch はチームによる共有計算リソースでのワークフロー構築を支援するために設計されています。これまでの例では `wandb launch` コマンドがローカルマシンでコンテナを同期的に実行していましたが、Launch queues と agents を使うことで、共有リソース上でのジョブの非同期実行や、優先度管理・ハイパーパラメーター最適化など高度な機能が利用できます。基本的な queue を作る手順：

1. [wandb.ai/launch](https://wandb.ai/launch) にアクセスし、**Create a queue** ボタンをクリック。
2. Queue を紐づける **Entity** を選択。
3. **Queue name** を入力。
4. **Resource** として **Docker** を選択。
5. **Configuration** 欄は現時点では空白のままでOK。
6. **Create queue** をクリック :rocket:

ボタンをクリックすると、ブラウザは queue ビューの **Agents** タブにリダイレクトされます。エージェントが稼働開始するまで、queue は **Not active** 状態で保持されます。

{{< img src="/images/launch/create_docker_queue.gif" alt="Docker queue creation" >}}

より詳細な queue 設定については、[高度な queue 設定ページ]({{< relref path="/launch/set-up-launch/setup-queue-advanced.md" lang="ja" >}})を参照してください。

## queue にエージェントを接続

queue ビューでは、ポーリングしているエージェントが一つもない場合、画面上部の赤いバナーに **Add an agent** ボタンが表示されます。このボタンをクリックすると、エージェント実行用コマンドが表示・コピーできます。コマンド例は次の通りです：

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

このコマンドをターミナルで実行すると、エージェントが起動し、指定された queue で実行待ちのジョブを監視します。ジョブが入ると、そのジョブのためのコンテナイメージをダウンロードまたはビルドし、ローカルで `wandb launch` コマンドを実行した場合と同様に実行します。

[Launch ページ](https://wandb.ai/launch)に戻り、queue が **Active** になっていることを確認できます。

## queue に job を投入

W&B アカウントで新しく作成した **launch-quickstart** プロジェクトに移動し、画面左側のナビゲーションから jobs タブを開きます。

**Jobs** ページには、これまで実行した runs から作成された W&B Jobs の一覧が表示されます。自分の launch job をクリックすると、ソースコードや依存関係、そのジョブから作成された runs などを確認できます。この walkthrough 完了後には、一覧に3つのジョブが存在します。

新しいジョブのいずれかを選択し、以下の手順で queue へ投入します：

1. **Launch** ボタンをクリックし、ジョブの queue への提出画面（Launch drawer）を表示します。
2. 先ほど作成した **Queue** を選択し、**Launch** をクリック。

これでジョブが queue へ送信されます。この queue を監視している agent がジョブを取得し、実行します。進捗は W&B UI またはエージェントのターミナル出力から確認できます。

`wandb launch` コマンドに `--queue` 引数を追加すれば、コマンドラインから直接ジョブを queue へ登録できます。たとえば、hello-world のコンテナジョブを queue へ投入するには次のコマンドを実行します：

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```