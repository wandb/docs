---
title: 'チュートリアル: プロジェクトから sweep ジョブを作成する'
description: 既存の W&B プロジェクトから sweep ジョブを作成する方法のチュートリアルです。
menu:
  default:
    identifier: existing-project
    parent: sweeps
---

このチュートリアルでは、既存の W&B Project から sweep ジョブを作成する方法を説明します。[Fashion MNIST データセット](https://github.com/zalandoresearch/fashion-mnist) を使用して、PyTorch の畳み込みニューラルネットワークで画像の分類方法を学習します。必要なコードやデータセットは、[W&B examples リポジトリ（PyTorch CNN Fashion）](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) にあります。

結果は [W&B Dashboard](https://app.wandb.ai/carey/pytorch-cnn-fashion) で確認できます。

## 1. Project の作成

まずはベースラインを作成しましょう。W&B examples の GitHub リポジトリから PyTorch MNIST データセットの example モデルをダウンロードします。次に、モデルをトレーニングします。トレーニングスクリプトは `examples/pytorch/pytorch-cnn-fashion` ディレクトリー内にあります。

1. リポジトリをクローン：`git clone https://github.com/wandb/examples.git`
2. この example を開く：`cd examples/pytorch/pytorch-cnn-fashion`
3. 手動で run を実行：`python train.py`

オプションで、この example が W&B App の UI ダッシュボードに表示されているのを確認できます。

[Project ページの例を見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. Sweep を作成

Project ページからサイドバーの [Sweep タブ]({{< relref "./sweeps-ui.md" >}}) を開き、**Create Sweep** を選択します。

{{< img src="/images/sweeps/sweep1.png" alt="Sweep overview" >}}

自動生成された設定は、これまで実行した run を元にスイープする値を推測します。どのハイパーパラメーターの範囲を試したいかを指定して設定を編集しましょう。Sweep をローンチすると、ホストされている W&B sweep server で新しいプロセスが開始されます。この中央サービスが、トレーニングジョブを実行するエージェント（マシン）を調整します。

{{< img src="/images/sweeps/sweep2.png" alt="Sweep configuration" >}}

## 3. エージェントのローンチ

次に、ローカルで agent をローンチしましょう。作業を分散して sweep ジョブをより早く完了させたい場合は、異なるマシンで最大 20 個までエージェントを同時にローンチできます。エージェントは次に試すパラメータのセットを表示します。

{{< img src="/images/sweeps/sweep3.png" alt="Launch agents" >}}

これで sweep が実行されています。以下の画像は、example sweep ジョブが実行中のダッシュボードの様子です。[Project ページの例を見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

{{< img src="/images/sweeps/sweep4.png" alt="Sweep dashboard" >}}

## 既存の run から新しい sweep をシード

以前にログした既存の run を使って、新しい sweep をローンチしましょう。

1. Project のテーブルを開きます。
2. テーブル左側のチェックボックスで使用したい run を選択します。
3. ドロップダウンから新しい sweep の作成をクリックします。

これで sweep がサーバー上に設定されます。あとは 1 つ以上のエージェントをローンチするだけで run の実行が始まります。

{{< img src="/images/sweeps/tutorial_sweep_runs.png" alt="Seed sweep from runs" >}}

{{% alert %}}
新しい sweep を bayesian sweep として実行する場合、選択した run も Gaussian Process のシードとして使われます。
{{% /alert %}}