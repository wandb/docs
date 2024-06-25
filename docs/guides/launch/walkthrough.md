---
description: W&B Launch のための開始ガイド
displayed_sidebar: default
---

import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Walkthrough

このページでは、W&B Launch ワークフローの基本を説明します。

:::tip
W&B Launch はコンテナ内で機械学習ワークロードを実行します。コンテナの知識は必須ではありませんが、このウォークスルーでは役立つかもしれません。コンテナについての基礎知識は [Docker ドキュメント](https://docs.docker.com/guides/docker-concepts/the-basics/what-is-a-container/) を参照してください。
:::

## Prerequisites

このウォークスルーには、Docker CLIとエンジンが動作するマシンへのターミナルアクセスが必要です。詳細については [Docker インストールガイド](https://docs.docker.com/engine/install/) を参照してください。

次のコマンドを使用して、Python 環境に `wandb>=0.17.1` をインストールします：

```bash
pip install --upgrade wandb
```

`wandb login` を実行するか、`WANDB_API_KEY` 環境変数を設定してW&Bに認証します。無料のWeights & Biasesアカウントにサインアップするには、[wandb.ai](https://wandb.ai) を訪れてください 🪄🐝

## Launch from an image

W&B に古典的なメッセージをログするシンプルなプレメイドコンテナを実行するには、ターミナルを開いて次のコマンドを実行します：

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart
```

前述のコマンドは `wandb/job_hello_world:main` コンテナイメージをダウンロードして実行します。Launch はコンテナを設定し、`launch-quickstart` プロジェクトにログされたすべての内容をW&Bに報告します。コンテナはW&Bにメッセージをログし、新しく作成された run へのリンクを表示します。リンクをクリックしてW&B UIで run を表示します。

## Launch from a git repository

[W&B Launch jobs リポジトリのソースコード](https://github.com/wandb/launch-jobs) から同じ hello-world ジョブを起動するには、次のコマンドを実行します：

```bash
wandb launch --uri https://github.com/wandb/launch-jobs.git --job-name hello-world-git --project launch-quickstart --build-context jobs/hello_world --dockerfile Dockerfile.wandb --entry-point "python job.py"
```
このコマンドは以下のことを行います：
1. [W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs) を一時ディレクトリにクローンします。
2. **hello** プロジェクトに **hello-world-git** という名前のジョブを作成します。このジョブは実行するコードの正確なソースコードと設定を追跡します。
3. `jobs/hello_world` ディレクトリと `Dockerfile.wandb` を使用してコンテナイメージをビルドします。
4. コンテナを開始し、`python job.py` を実行します。

コンソール出力には、イメージのビルドおよび実行が表示されます。コンテナの出力は前の例とほぼ同じであるはずです。

## Launch from local source code

Git リポジトリにバージョン管理されていないコードは、ローカルディレクトリパスを `--uri` 引数に指定することで起動できます。

空のディレクトリを作成し、次の内容の Python スクリプト `train.py` を追加します：

```python
import wandb

with wandb.init() as run:
    run.log({"hello": "world"})
```

次の内容の `requirements.txt` ファイルを追加します：

```text
wandb>=0.17.1
```

ディレクトリ内から次のコマンドを実行します：

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python train.py"
```

このコマンドは以下のことを行います：
1. 現在のディレクトリの内容を Code Artifact として W&B にログします。
2. **launch-quickstart** プロジェクトに **hello-world-code** という名前のジョブを作成します。
3. `train.py` および `requirements.txt` をベースイメージにコピーし、`pip install` で必要なパッケージをインストールしてコンテナイメージをビルドします。
4. コンテナを開始し、`python train.py` を実行します。

## Create a queue

Launch は Teams が共有計算リソースを使ってワークフローを構築するのを支援するよう設計されています。これまでの例では、`wandb launch` コマンドはローカルマシンで同期的にコンテナを実行していました。Launch のキューとエージェントを使用すると、共有リソース上でジョブを非同期に実行したり、優先順位設定やハイパーパラメーター最適化のような高度な機能を利用できます。基本的なキューを作成するには、次の手順に従います：

1. [wandb.ai/launch](https://wandb.ai/launch) に移動し、**Create a queue** ボタンをクリックします。
2. キューを関連付ける **Entity** を選択します。
3. **Queue name** を入力します。
4. **Resource** として **Docker** を選択します。
5. **Configuration** は今のところ空白のままにします。
6. **Create queue** ボタンをクリックします 🚀

ボタンをクリックすると、ブラウザはキュー表示の **Agents** タブにリダイレクトします。エージェントがポーリングを開始するまで、キューは **Not active** 状態のままです。

![](/images/launch/create_docker_queue.gif)

高度なキュー設定オプションについては、[高度なキュー設定ページ](./setup-queue-advanced.md)を参照してください。

## Connect an agent to the queue

ポーリングエージェントがない場合、キュー表示の上部に赤いバナーで **Add an agent** ボタンが表示されます。ボタンをクリックして、エージェントを実行するためのコマンドをコピーします。コマンドは以下のようになります：

```bash
wandb launch-agent --queue <queue-name> --entity <entity-name>
```

コマンドをターミナルで実行してエージェントを開始します。エージェントは指定されたキューをポーリングして実行するジョブを探します。ジョブを受信すると、エージェントはコンテナイメージをダウンロードまたはビルドしてジョブを実行します。これは `wandb launch` コマンドをローカルで実行した場合と同様です。

再び[Launchのページ](https://wandb.ai/launch) に戻り、キューが **Active** と表示されていることを確認します。

## Submit a job to the queue

W&B アカウントの **launch-quickstart** プロジェクトに移動し、画面左側のナビゲーションからジョブタブを開きます。

**Jobs** ページには、以前に実行された Runs から作成された W&B Jobs のリストが表示されます。ランチジョブをクリックして、ソースコード、依存関係、およびそのジョブから作成された任意の Runs を表示します。このウォークスルーを完了すると、リストに3つのジョブが表示されるはずです。

新しいジョブのうちの一つを選んで、以下の手順に従ってキューに送信します：

1. **Launch** ボタンをクリックしてジョブをキューに送信します。**Launch** ドロワーが表示されます。
2. 先ほど作成した **Queue** を選択し、**Launch** をクリックします。

これでジョブがキューに送信されます。このキューをポーリングしているエージェントがジョブを受け取り、実行します。ジョブの進行状況は W&B UI から、またはターミナルでエージェントの出力を確認することで監視できます。

`wandb launch` コマンドには `--queue` 引数を指定することで直接ジョブをキューにプッシュできます。例えば、hello-world コンテナジョブをキューに送信するには、次のコマンドを実行します：

```bash
wandb launch --docker-image wandb/job_hello_world:main --project launch-quickstart --queue <queue-name>
```