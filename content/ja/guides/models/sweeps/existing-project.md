---
title: 'Tutorial: Create sweep job from project'
description: 既存の W&B プロジェクトから sweep ジョブを作成する方法に関するチュートリアル。
menu:
  default:
    identifier: ja-guides-models-sweeps-existing-project
    parent: sweeps
---

このチュートリアルでは、既存の W&B プロジェクトからスイープジョブを作成する方法を説明します。[Fashion MNIST データセット](https://github.com/zalandoresearch/fashion-mnist)を使用して、PyTorch 畳み込みニューラルネットワークに画像を分類する方法を学習させます。必要なコードとデータセットは W&B リポジトリーにあります: [https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

この [W&B ダッシュボード](https://app.wandb.ai/carey/pytorch-cnn-fashion) で結果を確認してください。

## 1. プロジェクトの作成

まず、ベースラインを作成します。PyTorch MNIST データセットの例のモデルを W&B の例 GitHub レポジトリからダウンロードします。次に、モデルをトレーニングします。トレーニングスクリプトは `examples/pytorch/pytorch-cnn-fashion` ディレクトリーにあります。

1. このリポジトリをクローンします `git clone https://github.com/wandb/examples.git`
2. この例を開きます `cd examples/pytorch/pytorch-cnn-fashion`
3. 手動で run を実行します `python train.py`

オプションとして、W&B App UI ダッシュボードで例を確認することもできます。

[サンプルプロジェクトページを表示 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. Sweep の作成

あなたのプロジェクトページから、サイドバーの [Sweep タブ]({{< relref path="./sweeps-ui.md" lang="ja" >}}) を開き、**Create Sweep** を選択します。

{{< img src="/images/sweeps/sweep1.png" alt="" >}}

自動生成された設定は、完了した runs に基づいて sweep する値を推測します。設定を編集して、試したいハイパーパラメーターの範囲を指定してください。sweep をローンチすると、ホストされた W&B sweep サーバーで新しいプロセスが開始されます。この集中サービスは、トレーニングジョブを実行しているエージェント（マシン）を調整します。

{{< img src="/images/sweeps/sweep2.png" alt="" >}}

## 3. エージェントのローンチ

次に、エージェントをローカルでローンチします。最大 20 個のエージェントを異なるマシン上で並行してローンチすることができ、作業を分散してスイープジョブをより早く完了することができます。エージェントは次に試すパラメータのセットを出力します。

{{< img src="/images/sweeps/sweep3.png" alt="" >}}

これで、sweep を実行中です。以下の画像は、例のスイープジョブが実行されているときのダッシュボードの様子を示しています。[サンプルプロジェクトページを表示 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

{{< img src="/images/sweeps/sweep4.png" alt="" >}}

## 既存の runs で新しいスイープにシードを提供

以前にログした既存の runs を使用して、新しいスイープを開始します。

1. プロジェクトテーブルを開きます。
2. テーブルの左側にあるチェックボックスを使用して、使用したい runs を選択します。
3. ドロップダウンをクリックして、新しいスイープを作成します。

これでスイープが私たちのサーバーにセットアップされます。run を始めるには、エージェントを 1 つ以上ローンチするだけです。

{{< img src="/images/sweeps/tutorial_sweep_runs.png" alt="" >}}

{{% alert %}}
新しいスイープをベイジアンスイープとして開始する場合、選択した runs はガウスプロセスにもシードされます。
{{% /alert %}}