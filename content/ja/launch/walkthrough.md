---
title: 'チュートリアル: W&B Launch の基本'
description: W&B ローンンチのはじめかたガイド
menu:
  launch:
    identifier: walkthrough
    parent: launch
url: guides/launch/walkthrough
weight: 1
---

## Launch とは？

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

W&B Launch を使えば、トレーニング [runs]({{< relref "/guides/models/track/runs/" >}}) をデスクトップから Amazon SageMaker や Kubernetes などの計算リソースにかんたんにスケールできます。W&B Launch の設定が完了すれば、トレーニングスクリプトの実行、モデルの評価スイートの実施、プロダクション推論用モデルの準備などを、数クリックとコマンドだけで素早く行えます。

## 仕組み

Launch は、**launch jobs**、**queues**、**agents** の３つの基本要素で構成されています。

[*launch job*]({{< relref "./launch-terminology.md#launch-job" >}}) は、MLワークフロー上でタスクを設定・実行するための設計図です。launch job を作ったら、それを [*launch queue*]({{< relref "./launch-terminology.md#launch-queue" >}}) に追加できます。launch queue は「先入れ先出し（FIFO）」のキューで、指定した計算リソース（例えば Amazon SageMaker や Kubernetes クラスター）に job を送信できます。

job がキューに追加されると、[*launch agent*]({{< relref "./launch-terminology.md#launch-agent" >}}) がキューの中身をチェックし、キューで指定されたシステム上で job を実行します。

{{< img src="/images/launch/launch_overview.png" alt="W&B Launch overview diagram" >}}

ユースケースに応じて、自分またはチームメンバーが利用したい [計算リソースターゲット]({{< relref "./launch-terminology.md#target-resources" >}})（例: Amazon SageMaker）にあわせて launch queue を設定し、自身のインフラストラクチャー上に launch agent をデプロイします。

Launch の詳細な用語解説は [Terms and concepts]({{< relref "./launch-terminology.md" >}}) ページをご覧ください。

## 開始方法

ユースケースに応じて、W&B Launch に入門するには以下のリソースをぜひご活用ください。

* W&B Launch の使用が初めての場合は、[Launch walkthrough]({{< relref "#walkthrough" >}}) ガイドから始めることをおすすめします。
* [W&B Launch のセットアップ方法]({{< relref "/launch/set-up-launch/" >}}) を学ぶ。
* [launch job を作成する]({{< relref "./create-and-deploy-jobs/create-launch-job.md" >}})。
* よくあるタスク用のテンプレートとして、W&B Launch の [public jobs GitHub レポジトリ](https://github.com/wandb/launch-jobs) もご覧ください。例えば [Triton へのデプロイ](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton)、[LLM の評価](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) など、多様な例があります。
    * このレポジトリで作成された launch job を、公開されている [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) の W&B Project で確認できます。

## Walkthrough

このページでは W&B Launch ワークフローの基本を順を追って説明します。

{{% alert %}}
W&B Launch は、機械学習のワークロードをコンテナで実行します。コンテナに詳しくなくても問題ありませんが、この walkthrough では知識があるとより理解しやすくなります。コンテナについては [Docker ドキュメント](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/) をご参照ください。
{{% /alert %}}

## 事前準備

開始する前に、以下の要件を満たしていることをご確認ください。

1. https://wandb.ai/site でアカウント登録後、W&B アカウントにログインします。
2. この walkthrough では、Docker CLI とエンジンが動作するマシンへのターミナルアクセスが必要です。詳細は [Docker インストールガイド](https://docs.docker.com/engine/install/) をご覧ください。
3. W&B Python SDK バージョン `0.17.1` 以上をインストールします。
```bash
pip install wandb>=0.17.1
```
4. ターミナルで `wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定して W&B に認証します。

{{< tabpane text=true >}}
{{% tab "W&B にログイン" %}}
    ターミナルで次のコマンドを実行します:
    
    ```bash
    wandb login
    ```
{{% /tab %}}
{{% tab "環境変数を使う" %}}

    ```bash
    WANDB_API_KEY=<your-api-key>
    ```

    `<your-api-key>` をご自分の W&B API キーに置き換えてください。
{{% /tab %}}
{{% /tabpane %}}

## launch job を作成する
[launch job]({{< relref "./launch-terminology.md#launch-job" >}}) の作成は、Docker イメージ・git レポジトリ・ローカルソースコードのいずれか３つの方法から行えます。

{{< tabpane text=true >}}
{{% tab "Docker イメージを利用" %}}
W&B にメッセージをログするプリセットコンテナを実行するには、ターミナルで次のコマンドを実行してください。

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

このコマンドは `wandb/job_hello_world:main` のコンテナイメージをダウンロードして実行します。

Launch は、`wandb` で記録した内容が `launch-quickstart` プロジェクトにすべて報告されるようにコンテナを設定します。コンテナは W&B へメッセージをログし、新しい run へのリンクを表示します。そのリンクをクリックすると、W&B の UI で run を確認できます。
{{% /tab %}}
{{% tab "git レポジトリから" %}}
同じ hello-world job を [W&B Launch jobs レポジトリのソースコード](https://github.com/wandb/launch-jobs) から起動するには、次のコマンドを実行してください:

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git \\
--job-name hello-world-git --project launch-quickstart \\ 
--build-context jobs/hello_world --dockerfile Dockerfile.wandb \\ 
--entry-point "python job.py"
```
このコマンドは以下を行います:
1. [W&B Launch jobs レポジトリ](https://github.com/wandb/launch-jobs) を一時ディレクトリにクローンします。
2. **hello-world-git** という名前の job を **hello** プロジェクトに作成します。この job は実際に使用されたソースコードと設定を記録・追跡します。
3. `jobs/hello_world` ディレクトリと `Dockerfile.wandb` からコンテナイメージをビルドします。
4. コンテナを起動し、`job.py` の python スクリプトを実行します。

コンソール出力にはイメージのビルドと実行過程が表示され、先ほどの例とほぼ同じ結果が出力されます。

{{% /tab %}}
{{% tab "ローカルソースコードから" %}}

git レポジトリでバージョン管理されていないコードでも、`--uri` 引数にローカルディレクトリのパスを指定すれば launch できます。

空のディレクトリを作成し、`train.py` という Python スクリプトを以下の内容で作成してください。

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

さらに `requirements.txt` ファイルも以下の内容で作成します。

```text
wandb>=0.17.1
```

そのディレクトリ内で次のコマンドを実行します。

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

このコマンドは以下を行います:
1. カレントディレクトリの内容を W&B の Code Artifact としてログに記録します。
2. **hello-world-code** という名前の job を **launch-quickstart** プロジェクトに作成します。
3. `train.py` と `requirements.txt` をベースイメージにコピーしてコンテナイメージをビルドし、`pip install` で依存関係をインストールします。
4. コンテナを起動して `python train.py` を実行します。
{{% /tab %}}
{{< /tabpane >}}

## queue を作成する

Launch は、チームで共有リソースを活用したワークフロー構築をサポートするための設計です。これまでの例では、`wandb launch` コマンドはローカルマシン上ですぐにコンテナを実行してきました。Launch の queues や agents を使えば、共有リソース上でジョブを非同期実行したり、優先順位付けやハイパーパラメータ最適化などの高度な機能に対応できます。基本の queue を作成するには以下の手順をお試しください。

1. [wandb.ai/launch](https://wandb.ai/launch) にアクセスし、**Create a queue** ボタンをクリックします。
2. **Entity** を選択します。
3. **Queue name** を入力します。
4. **Resource** で **Docker** を選択します。
5. **Configuration** は今は空白のままで構いません。
6. **Create queue** をクリックします :rocket:

ボタンをクリックすると、ブラウザは queue ビューの **Agents** タブにリダイレクトされます。queue はエージェントのポーリングが開始されるまで **Not active** 状態のままです。

{{< img src="/images/launch/create_docker_queue.gif" alt="Docker queue creation" >}}

より高度な queue 設定については、[高度なキューのセットアップ]({{< relref "/launch/set-up-launch/setup-queue-advanced.md" >}})ページをご覧ください。

## agent を queue に接続する

queue ビューの画面上部の赤いバナーには、ポーリングエージェントが存在しない場合 **Add an agent** ボタンが表示されます。クリックすると、エージェントを立ち上げるコマンドをコピーできるようになります。コマンド例は以下のようになります。

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

ターミナルでこのコマンドを実行してエージェントを起動しましょう。エージェントは指定された queue を監視し、job が来ると受信してコンテナイメージをダウンロードあるいはビルドし、まるで `wandb launch` コマンドをローカルで実行した時のように job を実行します。

[Launch ページ](https://wandb.ai/launch) に戻り、queue の状態が **Active** になったことを確認します。

## queue に job を送信する

W&B アカウントで新しく作成した **launch-quickstart** プロジェクトに移動し、画面左側のナビゲーションから jobs タブを開きます。

**Jobs** ページには、これまで実行した run から作成された W&B Jobs の一覧が表示されます。自分の launch job をクリックすると、そのソースコードや依存関係、job から作成された run などを確認できます。この walkthrough 完了時点でリストには 3 つの jobs が表示されているはずです。

新しい jobs の中から 1 つを選び、以下の手順で queue に送信しましょう。

1. **Launch** ボタンをクリックして job をキューに送信します。すると **Launch** のドロワーが表示されます。
2. 先ほど作成した **Queue** を選択し、**Launch** をクリックします。

これで job は queue に送信されます。この queue を監視している agent が job を受け取り実行します。job の進捗は W&B UI から、あるいは agent のターミナル出力からも確認できます。

`wandb launch` コマンドに `--queue` 引数を指定すれば、job を直接 queue に投入することもできます。例えば hello-world のコンテナ job を queue に送信するには次のコマンドを実行します。

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```