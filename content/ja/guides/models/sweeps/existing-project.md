---
title: 'Tutorial: Create sweep job from project'
description: 既存の W&B プロジェクトから sweep ジョブを作成する方法のチュートリアル。
menu:
  default:
    identifier: ja-guides-models-sweeps-existing-project
    parent: sweeps
---

このチュートリアルでは、既存の W&B の **project** から **sweep** ジョブを作成する方法について説明します。ここでは、[Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) を使用して、PyTorch の畳み込みニューラルネットワークに画像の分類方法を学習させます。必要な **code** と **dataset** は、W&B のリポジトリ ([https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)) にあります。

[W&B **Dashboard**](https://app.wandb.ai/carey/pytorch-cnn-fashion) で **result** を確認してください。

## 1. **Project** を作成する

まず、**baseline** を作成します。W&B の examples GitHub リポジトリから PyTorch MNIST **dataset** のサンプル **model** をダウンロードします。次に、**model** を **training** します。**training script** は `examples/pytorch/pytorch-cnn-fashion` ディレクトリー内にあります。

1. このリポジトリをクローンします: `git clone https://github.com/wandb/examples.git`
2. このサンプルを開きます: `cd examples/pytorch/pytorch-cnn-fashion`
3. 手動で **run** を実行します: `python train.py`

オプションで、W&B App UI **dashboard** に表示されるサンプルを確認します。

[サンプル **project** ページを表示 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. **Sweep** を作成する

**project** ページから、サイドバーの [**Sweep** タブ]({{< relref path="./sweeps-ui.md" lang="ja" >}}) を開き、**Create Sweep** を選択します。

{{< img src="/images/sweeps/sweep1.png" alt="" >}}

自動生成された **configuration** は、完了した **run** に基づいて、**sweep** する **value** を推測します。**試したい hyperparameter** の範囲を指定するように **configuration** を編集します。**sweep** を **Launch** すると、ホストされている W&B **sweep server** 上で新しい **process** が開始されます。この集中型サービスは、**training** ジョブを実行しているマシンである **agent** を調整します。

{{< img src="/images/sweeps/sweep2.png" alt="" >}}

## 3. **Agent** を **Launch** する

次に、ローカルで **agent** を **Launch** します。**work** を分散して **sweep** ジョブをより迅速に完了したい場合は、最大 20 個の **agent** を異なるマシン上で並行して **Launch** できます。**agent** は、次に試す **parameter** のセットを出力します。

{{< img src="/images/sweeps/sweep3.png" alt="" >}}

これで **sweep** が実行されました。次の画像は、サンプル **sweep** ジョブの実行中の **dashboard** の外観を示しています。[サンプル **project** ページを表示 →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

{{< img src="/images/sweeps/sweep4.png" alt="" >}}

## 既存の **run** で新しい **sweep** をシードする

以前に **log** した既存の **run** を使用して、新しい **sweep** を **Launch** します。

1. **project** テーブルを開きます。
2. テーブルの左側にあるチェックボックスを使用して、使用する **run** を選択します。
3. ドロップダウンをクリックして、新しい **sweep** を作成します。

これで、**sweep** が当社の **server** 上にセットアップされます。**run** の実行を開始するには、1 つ以上の **agent** を **Launch** するだけです。

{{< img src="/images/sweeps/tutorial_sweep_runs.png" alt="" >}}

{{% alert %}}
新しい **sweep** をベイズ **sweep** として開始すると、選択された **run** もガウス **process** をシードします。
{{% /alert %}}
