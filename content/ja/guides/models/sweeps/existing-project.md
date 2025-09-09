---
title: 'チュートリアル: Project から sweep ジョブを作成する'
description: 'チュートリアル: 既存の W&B Project から sweep ジョブを作成する方法。'
menu:
  default:
    identifier: ja-guides-models-sweeps-existing-project
    parent: sweeps
---

このチュートリアルでは、既存の W&B Project から sweep ジョブを作成する方法を説明します。PyTorch の畳み込みニューラルネットワークを使って、[Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) で画像の分類を学習させます。必要なコードとデータセットは [W&B examples repository (PyTorch CNN Fashion)](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) にあります。

結果はこの [W&B Dashboard](https://app.wandb.ai/carey/pytorch-cnn-fashion) で確認できます。

## 1. Create a project

まずベースラインを作成します。W&B examples の GitHub リポジトリーから PyTorch の MNIST データセット用サンプルモデルをダウンロードし、続いてモデルをトレーニングします。トレーニングスクリプトは `examples/pytorch/pytorch-cnn-fashion` ディレクトリーにあります。

1. このリポジトリーをクローン `git clone https://github.com/wandb/examples.git`
2. このサンプルを開く `cd examples/pytorch/pytorch-cnn-fashion`
3. 手動で Run を実行する `python train.py`

必要に応じて、この例を W&B App の UI ダッシュボードで確認できます。

[サンプルの Project ページを見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. Create a sweep

Project ページから、サイドバーの [Sweep タブ]({{< relref path="./sweeps-ui.md" lang="ja" >}}) を開き、**Create Sweep** を選択します。

{{< img src="/images/sweeps/sweep1.png" alt="sweep の概要" >}}

自動生成された設定は、完了済みの Runs に基づいて sweep 対象とする値を推測します。試したいハイパーパラメーターの範囲を指定するように設定を編集してください。sweep を開始すると、ホストされた W&B の sweep server 上で新しいプロセスが起動します。この集中管理されたサービスが、トレーニングジョブを実行するマシンであるエージェントを調整します。

{{< img src="/images/sweeps/sweep2.png" alt="sweep の設定" >}}

## 3. Launch agents

次に、ローカルでエージェントを起動します。作業を分散して sweep ジョブをより早く終わらせたい場合は、最大 20 個のエージェントを異なるマシンで並行起動できます。エージェントは、次に試すパラメータのセットを出力します。

{{< img src="/images/sweeps/sweep3.png" alt="エージェントを起動" >}}

これで sweep が実行中です。以下の画像は、このサンプルの sweep ジョブを実行中のダッシュボードの様子です。[サンプルの Project ページを見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

{{< img src="/images/sweeps/sweep4.png" alt="sweep のダッシュボード" >}}

## Seed a new sweep with existing runs

以前にログした既存の Runs を使って新しい sweep を開始します。

1. Project のテーブルを開きます。
2. テーブル左側のチェックボックスで使用したい Runs を選択します。
3. ドロップダウンをクリックして新しい sweep を作成します。

sweep は当社のサーバー上にセットアップされます。あとは 1 つ以上のエージェントを起動して Runs の実行を開始するだけです。

{{< img src="/images/sweeps/tutorial_sweep_runs.png" alt="Runs から sweep をシードする" >}}

{{% alert %}}
新しい sweep を Bayesian sweep として開始すると、選択した Runs はガウス過程のシードにもなります。
{{% /alert %}}