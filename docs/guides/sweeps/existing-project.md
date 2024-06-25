---
description: 既存の W&B プロジェクトから sweep ジョブを作成する方法のチュートリアル。
displayed_sidebar: default
---


# Tutorial - Create sweeps from existing projects

<head>
    <title>Create sweeps from existing projects Tutorial</title>
</head>

このチュートリアルでは、既存の W&B Project から Sweep ジョブを作成する手順を説明します。[Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) を使用して、PyTorch の畳み込みニューラルネットワークで画像を分類する方法を学びます。必要なコードとデータセットは、W&B のリポジトリーにあります: [https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

結果をこの [W&B Dashboard](https://app.wandb.ai/carey/pytorch-cnn-fashion) で確認してください。

## 1. プロジェクトを作成する

まず、ベースラインを作成します。W&B の GitHub リポジトリーから PyTorch MNIST データセットのサンプルモデルをダウンロードし、モデルをトレーニングします。トレーニングスクリプトは `examples/pytorch/pytorch-cnn-fashion` ディレクトリー内にあります。

1. このリポジトリーをクローンします `git clone https://github.com/wandb/examples.git`
2. この例を開きます `cd examples/pytorch/pytorch-cnn-fashion`
3. 手動で run を実行します `python train.py`

オプションとして、W&B アプリの UI ダッシュボードでこの例を確認できます。

[例のプロジェクトページを見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. Sweep を作成する

[プロジェクトページ](../app/pages/project-page.md) から、サイドバーの [Sweep タブ](./sweeps-ui.md) を開き、**Create Sweep** を選択します。

![](@site/static/images/sweeps/sweep1.png)

自動生成された設定は、実行した run に基づいて Sweeps する値を推測します。設定を編集して、試したいハイパーパラメーターの範囲を指定します。sweep を開始すると、ホストされている W&B sweep server 上に新しいプロセスが開始されます。この集中型サービスは、トレーニングジョブを実行しているエージェント（マシン）を調整します。

![](@site/static/images/sweeps/sweep2.png)

## 3. エージェントを起動する

次に、エージェントをローカルで起動します。最大 20 台のマシンで並行してエージェントを起動でき、作業を分散させて Sweep ジョブをより速く完了できます。エージェントは次に試すパラメータのセットを出力します。

![](@site/static/images/sweeps/sweep3.png)

これで Sweep を実行している状態です。以下の画像は、例の Sweep ジョブが実行されているときのダッシュボードの様子を示しています。[例のプロジェクトページを見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

![](https://paper-attachments.dropbox.com/s\_5D8914551A6C0AABCD5718091305DD3B64FFBA192205DD7B3C90EC93F4002090\_1579066494222\_image.png)

## 既存の run で新しい sweep を開始する

以前にログを記録した既存の run を使用して、新しい sweep を開始します。

1. プロジェクトテーブルを開きます。
2. テーブル左側のチェックボックスで使用したい run を選択します。
3. 新しい sweep を作成するドロップダウンをクリックします。

これで sweep はサーバー上に設定されます。run を開始するには、1 台以上のエージェントを起動するだけです。

![](/images/sweeps/tutorial_sweep_runs.png)

:::info
新しい sweep をベイズ sweep として開始すると、選択された run はガウス過程もシードします。
:::