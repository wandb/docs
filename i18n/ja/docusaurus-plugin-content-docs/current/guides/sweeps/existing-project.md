---
description: Tutorial on how to create sweep jobs from a pre-existing Weights & Biases project.
---

# チュートリアル - 既存プロジェクトからスウィープを作成

<head>
    <title>Create sweeps from existing projects Tutorial</title>
</head>

前述のチュートリアルは、既存のWeights & Biasesプロジェクトからスウィープジョブを作成する方法に関するステップをガイドします。[Fashion MNISTデータセット](https://github.com/zalandoresearch/fashion-mnist)を使って PyTorchの畳み込みニューラルネットワークの画像の分類方法をトレーニングします。データセットがWeights & Biasesレポジトリにある場所で必要なコード： [https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

この[W&Bダッシュボード](https://app.wandb.ai/carey/pytorch-cnn-fashion)で結果を探索します。

## 1. プロジェクトを作成する​

まずベースラインを作成します。PyTorch MNISTデータセットサンプルモデルを、Weights & BiasesサンプルGitHubレポジトリからダウンロードします。次にモデルのトレーニングを行います。トレーニングスクリプトは、`examples/pytorch/pytorch-cnn-fashion`ディレクトリーにあります。

1. このレポジトリ`git clone`を複製します `git clone https://github.com/wandb/examples.git`
2. このサンプルcdを開きます  `cd examples/pytorch/pytorch-cnn-fashion`
3. Runを手動で実行します `python train.py`

オプションで、W&BアプリUIダッシュボードに表示された例を探索します。

[サンプルプロジェクトページを表示  →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. スウィープを作成する​

[プロジェクトページ](../app/pages/project-page.md)から、サイドバーにある[`Sweep（スウィープ）`タブ](./sweeps-ui.md) を開き、`Create Sweep（スウィープを作成）`を選びます。

![](@site/static/images/sweeps/sweep1.png)

自動生成された設定は、完了したunsに基づいて、スウィープする値を推測します。設定を編集して、試したいハイパーパラメーターの範囲を指定します。スウィープを起動すると、ホスティングされるW&Bスウィープサーバー上で新しいプロセスが開始します。この集中型サービスは、エージェントと、トレーニングジョブを実行しているマシンを調整します。

![](@site/static/images/sweeps/sweep2.png)

## 3. エージェントを起動する​

次に、エージェントをローカルで起動します。作業を分散てスウィープジョブをより迅速に終了したい場合は、複数のマシン上で最大20のエージェントを並行して起動できます。エージェントは、次に試しているパラメーターのセットをプリントアウトします。

![](@site/static/images/sweeps/sweep3.png)

現在スウィープを実行しています。以下の画像は、サンプルスウィープジョブ実行中のダッシュボードの様子を示しています。[サンプルプロジェクトページを表示 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

![](https://paper-attachments.dropbox.com/s\_5D8914551A6C0AABCD5718091305DD3B64FFBA192205DD7B3C90EC93F4002090\_1579066494222\_image.png)

## 既存のrunで新しいスウィープのシードを設定する​

以前記録した既存のrunを使って新しいスウィープを開始します。

1. プロジェクトテーブルを開きます。
2. テーブルの左側にあるチェックボックスで、使用したいrunを選択します。
3. ドロップダウンをクリックして新しいスウィープを作成します。

スウィープは弊社のサーバー上ではセットアップされません。必要なことは、1つまたは複数のエージェントを開始して、runの実行を開始するだけです。

![](/images/sweeps/tutorial_sweep_runs.png)

:::info
新しいスウィープをベイズスウィープとして開始する場合、選択したrunは、ガウスプロセスのシード設定も行います。
:::


