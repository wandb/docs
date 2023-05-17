---
description: W&B Launchのクイックスタートガイド。
---

# クイックスタート

このガイドに従って、W&B Launchの使用を開始してください。このガイドでは、**ジョブ**、**ローンチキュー**、**ローンチキュー**というローンチワークフローの基本的な構成要素のセットアップを説明します。具体的には、ニューラルネットワークをトレーニングするジョブを作成し、ローカルマシンでジョブを実行するために使用されるローンチキューを作成し、次にキューをポーリングし、Dockerを使用してキューからポップされたジョブを実行するローンチエージェントを作成します。

## 始める前に
始める前に、ローンチエージェントを実行するマシンにDockerをインストールしてください。Dockerのインストール方法については、[Dockerのドキュメント](https://docs.docker.com/get-docker/)を参照してください。また、続行する前に、Dockerデーモンがマシンで実行されていることを確認してください。

## ジョブを作成する

ジョブは、関連するソースコードがあるW&Bのrunから自動的に作成されます。runにソースコードを関連付ける方法の詳細については、[このドキュメント](create-job.md)を参照してください。

以下のPythonコードをコピーして、`train.py`という名前のファイルに保存します。

```python
import wandb

config = {
    "epochs": 10
}

with wandb.init(config=config):
    config = wandb.config
    for epoch in range(1, config.epochs):
        loss = config.epochs / epoch
        accuracy = (1 + (epoch / config.epochs))/2
        wandb.log({
            "loss": loss, 
            "accuracy": accuracy, 
            "epoch": epoch})
    wandb.run.log_code()
```
依存関係をインストールし、スクリプトを実行するには、端末で次のコマンドを実行してください：

```bash
pip install wandb>=0.14.0
python train.py
```

スクリプトが完了するまで実行させ、次のステップに進んでください。コンソールの出力はおおよそ以下のようになります：

```
wandb: Syncing run trim-planet-100
wandb: ⭐️ プロジェクトを表示します https://wandb.ai/bcanfieldsherman/uncategorized
wandb: 🚀 runを表示します https://wandb.ai/bcanfieldsherman/uncategorized/runs/5av9db29
wandb: W&B プロセスが終了するのを待っています... (成功)。
wandb: 
wandb: Run 履歴：
wandb: accuracy ▁▂▃▄▅▅▆▇█
wandb:    エポック ▁▂▃▄▅▅▆▇█
wandb:     loss █▄▃▂▂▁▁▁▁
wandb: 
wandb: Run summary：
wandb: accuracy 0.95
wandb:    エポック 9
wandb:     loss 1.11111
wandb: 
wandb: 🚀 run trim-planet-100を表示します： https://wandb.ai/bcanfieldsherman/uncategorized/runs/5av9db29
wandb: 同期済み 4 W&B ファイル, 0 メディアファイル, 9 アーティファクトファイル, 1 その他ファイル
```
新しい**launch-quickstart**プロジェクトに移動し、画面左側のナビゲーションからジョブタブを開いてください。

![](/images/launch/jobs-tab.png)

**Jobs**ページには、以前に実行されたW&B Runsから作成されたW&B Jobsのリストが表示されます。**job-source-launch-quickstart-train.py:v0**という名前のジョブが表示されるはずです。ジョブページからジョブの名前を編集して、覚えやすい名前に変更することもできます。ジョブをクリックして、ジョブのソースコードや依存関係、このジョブから起動されたrunsのリストが含まれる詳細なビューを開きます。

## ジョブをキューに入れる

ジョブのページに戻ってください。以下の画像のような表示になっているはずです。

![](/images/launch/simple-job.png)

上部右側の**Launch**ボタンをクリックして、このジョブから新しいrunを起動します。画面右側からドロワーがスライドして出てきて、新しいrunのいくつかのオプションが表示されます。

* **Job version**: 起動するジョブのバージョン。Jobsは他のW&Bアーティファクトと同様にバージョン管理されます。 ジョブを実行するために使用されるソフトウェア依存関係やソースコードに変更が加えられた場合は、同じジョブの異なるバージョンが作成されます。今回は1つのバージョンしかないので、デフォルトの**@latest**バージョンを選択します。
* **Overrides**: ジョブの入力の新しい値。これらは、エントリポイントのコマンド、引数、または新しいrunの`wandb.config`内の値を変更するために使用できます。私たちのrunは`wandb.config`で`epochs`という1つの値を持っていました。これをオーバーライドフィールドで上書きすることができます。また、** Paste from...**ボタンをクリックすることで、他のrunsから値を貼り付けることもできます。
* **Queue**: runを起動するキュー。まだキューを作成していない場合、**Starter Queue**を作成するオプションがあるはずです。このキューは、Dockerを使用してローカルマシンでrunsを起動するために使用されます。

![](/images/launch/starter-launch.gif)

ジョブを必要に応じて設定したら、ドロワーの下部にある**launch now**ボタンをクリックして、launchジョブをキューに入れてください。

## ランチエージェントを開始する

ジョブを実行するために、起動キューで待機しているランチエージェントを開始する必要があります。

1. [wandb.ai/launch](https://wandb.ai/launch)から、起動キューのページに移動します。
2. **Add an agent**ボタンをクリックします。
3. W&B CLIコマンドが記載されたモーダルが表示されます。これをコピーして、ターミナルに貼り付けて実行してください。

![](/images/launch/activate_starter_queue_agent.png)

一般的に、launchエージェントを開始するコマンドは次のとおりです。

```bash
wandb launch-agent -e <entity-name> -q <queue-name>
```

ターミナル内で、エージェントがキューをポーリングするのを見ることができます。エージェントは、以前にエンキューしたジョブを検出し、実行を開始します。まず、エージェントは選択したジョブバージョンからコンテナイメージを構築します。次に、エージェントは `docker run` を使って、ローカルホストでジョブを実行します。

それだけです！Launchワークスペースまたはターミナルにアクセスして、launchジョブのステータスを確認してください。ジョブは、先入れ先出しの順序（FIFO）で実行されます。キューにプッシュされたすべてのジョブは、同じリソースタイプとリソース引数を使用します。