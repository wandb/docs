---
description: >-
  Tutorial on how to create sweep jobs from a pre-existing Weights & Biases
  project.
displayed_sidebar: default
---

# チュートリアル - 既存のプロジェクトからスイープを作成する

<head>
    <title>既存のプロジェクトからスイープを作成するチュートリアル</title>
</head>

このチュートリアルでは、既存のWeights & Biasesプロジェクトからスイープジョブを作成する方法を説明します。PyTorchの畳み込みニューラルネットワークを使用して、[Fashion MNISTデータセット](https://github.com/zalandoresearch/fashion-mnist)をトレーニングし、画像の分類方法を学習します。必要なコードとデータセットは、Weights & Biasesのリポジトリにあります：[https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

この[W&Bダッシュボード](https://app.wandb.ai/carey/pytorch-cnn-fashion)で結果を確認してください。

## 1. プロジェクトを作成する

まず、ベースラインを作成します。Weights & Biasesのexamples GitHubリポジトリからPyTorch MNISTデータセットのサンプルモデルをダウンロードします。次に、モデルをトレーニングします。トレーニングスクリプトは、`examples/pytorch/pytorch-cnn-fashion`ディレクトリ内にあります。

1. このリポジトリをクローンする `git clone https://github.com/wandb/examples.git`
2. この例を開く `cd examples/pytorch/pytorch-cnn-fashion`
3. runを手動で実行する `python train.py`

オプションで、W&BアプリのUIダッシュボードで例を確認します。

[プロジェクトページの例を見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. スイープを作成する

[プロジェクトページ](../app/pages/project-page.md)から、サイドバーの[スイープタブ](./sweeps-ui.md)を開き、**スイープを作成**を選択します。

![](@site/static/images/sweeps/sweep1.png)
自動生成された設定では、完了したrunに基づいてスイープ対象の値を推測します。設定を編集して、試したいハイパーパラメータの範囲を指定します。スイープを開始すると、ホストされたW&Bスイープサーバーで新しいプロセスが開始されます。この一元化されたサービスは、トレーニングジョブを実行しているマシンであるエージェントを調整します。

![](@site/static/images/sweeps/sweep2.png)

## 3. エージェントの起動

次に、エージェントをローカルで起動します。分散して作業を行い、スイープジョブをより迅速に終了させたい場合、異なるマシンで最大20のエージェントを同時に起動できます。エージェントは、次に試すパラメータのセットを出力します。

![](@site/static/images/sweeps/sweep3.png)

これでスイープが実行されています。次の画像は、ダッシュボードが例示されたスイープジョブが実行されている時の様子です。[例のプロジェクトページを表示 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

![](https://paper-attachments.dropbox.com/s_5D8914551A6C0AABCD5718091305DD3B64FFBA192205DD7B3C90EC93F4002090_1579066494222_image.png)

## 既存のrunで新しいスイープをシードする

以前にログした既存のrunを使用して、新しいスイープを開始します。

1. プロジェクトテーブルを開きます。
2. テーブルの左側にあるチェックボックスで、使用するrunを選択します。
3. ドロップダウンをクリックして、新しいスイープを作成します。

スイープはこれでサーバー上に設定されます。あとは、1つ以上のエージェントを起動してrunを開始するだけです。

![](/images/sweeps/tutorial_sweep_runs.png)

:::info
新しいスイープをベイズ探索として開始する場合、選択されたrunはガウス過程もシードします。
:::