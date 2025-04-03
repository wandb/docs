---
title: 'Tutorial: Create sweep job from project'
description: 既存の W&B プロジェクトから sweep ジョブを作成する方法のチュートリアル。
menu:
  default:
    identifier: ja-guides-models-sweeps-existing-project
    parent: sweeps
---

このチュートリアルでは、既存の W&B の プロジェクト から sweep job を作成する方法について説明します。ここでは、[Fashion MNIST データセット](https://github.com/zalandoresearch/fashion-mnist) を使用して、PyTorch の畳み込み ニューラルネットワーク に画像の分類方法を学習させます。必要な コード と データセット は、W&B のリポジトリにあります。[https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

この [W&B ダッシュボード](https://app.wandb.ai/carey/pytorch-cnn-fashion) で 結果 を確認してください。

## 1. プロジェクト を作成する

まず、 ベースライン を作成します。W&B examples GitHub リポジトリ から PyTorch MNIST データセット のサンプル モデル をダウンロードします。次に、 モデル を トレーニング します。トレーニング スクリプト は、`examples/pytorch/pytorch-cnn-fashion` ディレクトリー 内にあります。

1. このリポジトリをクローンします `git clone https://github.com/wandb/examples.git`
2. このサンプルを開きます `cd examples/pytorch/pytorch-cnn-fashion`
3. run を手動で実行します `python train.py`

オプションで、W&B App UI ダッシュボード に表示される例を確認してください。

[プロジェクト ページの例を見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. sweep を作成する

プロジェクト ページから、サイドバーの [Sweep tab]({{< relref path="./sweeps-ui.md" lang="ja" >}}) を開き、**Create Sweep** を選択します。

{{< img src="/images/sweeps/sweep1.png" alt="" >}}

自動生成された 設定 は、完了した run に基づいて sweep する 値 を推測します。試したい ハイパーパラメーター の範囲を指定するには、 設定 を編集します。sweep を 起動 すると、ホストされている W&B sweep server で新しい プロセス が開始されます。この集中型 サービス は、 トレーニング job を実行している マシン である エージェント を調整します。

{{< img src="/images/sweeps/sweep2.png" alt="" >}}

## 3. エージェント を 起動 する

次に、 ローカル で エージェント を 起動 します。作業を分散して sweep job をより迅速に完了したい場合は、最大 20 個の エージェント を異なる マシン で 並行 して 起動 できます。エージェント は、次に試す パラメータ のセットを出力します。

{{< img src="/images/sweeps/sweep3.png" alt="" >}}

これで sweep が実行されました。次の図は、サンプル sweep job の実行中に ダッシュボード がどのように表示されるかを示しています。[プロジェクト ページの例を見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

{{< img src="/images/sweeps/sweep4.png" alt="" >}}

## 既存の run で新しい sweep をシードする

以前に ログ に記録した既存の run を使用して、新しい sweep を 起動 します。

1. プロジェクト テーブル を開きます。
2. テーブル の左側にある チェックボックス で、使用する run を選択します。
3. ドロップダウン をクリックして、新しい sweep を作成します。

これで、sweep が サーバー 上にセットアップされます。run の実行を開始するには、1 つ以上の エージェント を 起動 するだけです。

{{< img src="/images/sweeps/tutorial_sweep_runs.png" alt="" >}}

{{% alert %}}
新しい sweep を ベイジアン sweep として開始すると、選択した run によって ガウス 過程 もシードされます。
{{% /alert %}}
