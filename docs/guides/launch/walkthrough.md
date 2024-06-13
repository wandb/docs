---
description: W&B Launch の開始ガイド。
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Walkthrough

このページでは、W&B Launch ワークフローの基本について説明します。

:::tip
W&B Launch はコンテナ内で機械学習のワークロードを実行します。コンテナに関する知識は必須ではありませんが、このウォークスルーに役立つかもしれません。コンテナの基本については [Docker documentation](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/) を参照してください。
:::

## Prerequisites

このウォークスルーには、Docker CLI とエンジンが動作するマシンへの端末アクセスが必要です。詳細については [Docker installation guide](https://docs.docker.com/engine/install/) を参照してください。

次のコマンドを使用して、Python 環境に `wandb>=0.17.1` をインストールします:

```bash
pip install --upgrade wandb
```

`wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定して W&B に認証します。無料の Weights & Biases アカウントにサインアップするには、[wandb.ai](https://wandb.ai) を訪問してください 🪄🐝

## Launch from an image

W&B にクラシックなメッセージをログするシンプルなプリメイドコンテナを実行するには、端末を開いて次のコマンドを実行します:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

上記のコマンドは、コンテナイメージ `wandb/job_hello_world:main` をダウンロードして実行します。Launch はコンテナを設定し、`wandb` でログされたすべての情報を `launch-quickstart` プロジェクトに報告します。コンテナは W&B にメッセージをログし、新しく作成された run へのリンクを表示します。リンクをクリックして W&B UI で run を表示します。

## Launch from a git repository

[W&B Launch jobs repository のソースコード](https://github.com/wandb/launch-jobs) から同じ hello-world ジョブを起動するには、次のコマンドを実行します:

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git --job-name hello-world-git --project launch-quickstart --build-context jobs/hello_world --dockerfile Dockerfile.wandb --entry-point "python job.py"
```
このコマンドは以下を行います:
1. [W&B Launch jobs repository](https://github.com/wandb/launch-jobs) を一時ディレクトリーにクローンします。
2. **hello** プロジェクトに **hello-world-git** という名前のジョブを作成します。このジョブは、コードを実行するために使用された正確なソースコードと設定を追跡します。
3. `jobs/hello_world` ディレクトリーと `Dockerfile.wandb` からコンテナイメージをビルドします。
4. コンテナを起動し、`python job.py` を実行します。

コンソール出力には、イメージのビルドと実行が表示されます。コンテナの出力は前の例とほぼ同じであるはずです。

## Launch from local source code

Git リポジトリーでバージョン管理されていないコードは、`--uri` 引数にローカルディレクトリーパスを指定して起動できます。

空のディレクトリーを作成し、次の内容の Python スクリプト `train.py` を追加します:

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

次の内容の `requirements.txt` ファイルを追加します:

```text
wandb>=0.17.1
```

ディレクトリー内から次のコマンドを実行します:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

このコマンドは以下を行います:
1. 現在のディレクトリーの内容を W&B に Code Artifact としてログします。
2. **launch-quickstart** プロジェクトに **hello-world-code** という名前のジョブを作成します。
3. `train.py` と `requirements.txt` をベースイメージにコピーし、`pip install` で必要なものをインストールしてコンテナイメージをビルドします。
4. コンテナを起動し、`python train.py` を実行します。

## Create a queue

Launch は、共有コンピュートリソースを中心にワークフローを構築するために設計されています。これまでの例では、`wandb launch` コマンドはローカルマシンで同期的にコンテナを実行していました。Launch のキューとエージェントを使用すると、ジョブを共有リソースで非同期に実行したり、優先順位付けやハイパーパラメーター最適化などの高度な機能を利用できます。基本的なキューを作成するには、次の手順に従います:

1. [wandb.ai/launch](https://wandb.ai/launch) に移動し、**Create a queue** ボタンをクリックします。
2. キューに関連付ける **Entity** を選択します。
3. **Queue name** を入力します。
4. **Resource** として **Docker** を選択します。
5. **Configuration** は今のところ空白のままにします。
6. **Create queue** をクリックします :rocket:

ボタンをクリックすると、ブラウザはキュー表示の **Agents** タブにリダイレクトされます。エージェントがポーリングを開始するまで、キューは **Not active** 状態のままです。

![](/images/launch/create_docker_queue.gif)

高度なキュー設定オプションについては、[advanced queue setup page](./setup-queue-advanced.md) を参照してください。

## Connect an agent to the queue

キューにポーリングエージェントがない場合、キュー表示の上部に赤いバナーで **Add an agent** ボタンが表示されます。ボタンをクリックして、エージェントを実行するためのコマンドをコピーします。コマンドは次のようになります:

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

端末でコマンドを実行してエージェントを起動します。エージェントは指定されたキューをポーリングしてジョブを実行します。受信すると、エージェントはジョブのためにコンテナイメージをダウンロードまたはビルドし、実行します。

[Launch ページ](https://wandb.ai/launch) に戻り、キューが **Active** と表示されていることを確認します。

## Submit a job to the queue

W&B アカウントの新しい **launch-quickstart** プロジェクトに移動し、画面左側のナビゲーションからジョブタブを開きます。

**Jobs** ページには、以前に実行された run から作成された W&B Jobs のリストが表示されます。launch ジョブをクリックして、ソースコード、依存関係、およびジョブから作成された run を表示します。このウォークスルーを完了すると、リストに 3 つのジョブが表示されるはずです。

新しいジョブのいずれかを選択し、次の手順に従ってキューに送信します:

1. **Launch** ボタンをクリックしてジョブをキューに送信します。**Launch** ドロワーが表示されます。
2. 先ほど作成した **Queue** を選択し、**Launch** をクリックします。

これでジョブがキューに送信されます。このキューをポーリングしているエージェントがジョブを取得して実行します。ジョブの進行状況は W&B UI から、または端末のエージェント出力を確認することで監視できます。

`wandb launch` コマンドは、`--queue` 引数を指定することでジョブを直接キューにプッシュできます。例えば、hello-world コンテナジョブをキューに送信するには、次のコマンドを実行します:

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```