---
title: チュートリアル：Project から sweep ジョブを作成する
description: 既存の W&B プロジェクトから sweep ジョブを作成する方法のチュートリアル。
menu:
  default:
    identifier: ja-guides-models-sweeps-existing-project
    parent: sweeps
---

このチュートリアルでは、既存の W&B Project から sweep ジョブを作成する方法を解説します。PyTorch の畳み込みニューラルネットワーク（CNN）を使って [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) の画像分類を行います。必要なコードとデータセットは [W&B examples repository (PyTorch CNN Fashion)](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion) にあります。

結果はこの [W&B Dashboard](https://app.wandb.ai/carey/pytorch-cnn-fashion) で確認できます。

## 1. Project の作成

まずはベースラインを作成しましょう。W&B examples の GitHub リポジトリから PyTorch MNIST データセットのモデル例をダウンロードし、モデルをトレーニングします。トレーニングスクリプトは `examples/pytorch/pytorch-cnn-fashion` ディレクトリー内にあります。

1. リポジトリをクローン： `git clone https://github.com/wandb/examples.git`
2. この例に移動： `cd examples/pytorch/pytorch-cnn-fashion`
3. 手動で run を実行： `python train.py`

オプションとして、W&B App UI のダッシュボードで例の Project の動作を確認できます。

[Project ページを見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. sweep の作成

Project ページからサイドバーの [Sweep タブ]({{< relref path="./sweeps-ui.md" lang="ja" >}}) を開き、**Create Sweep** を選択します。

{{< img src="/images/sweeps/sweep1.png" alt="Sweep overview" >}}

自動生成された設定では、これまでに完了した run に基づき、 sweep する値が推測されます。設定を編集し、試したいハイパーパラメータの範囲を指定しましょう。sweep をローンチすると、W&B のホストされた sweep サーバー上で新しいプロセスが開始されます。この中央サービスがトレーニングジョブを実行するエージェント（agent）たちを調整します。

{{< img src="/images/sweeps/sweep2.png" alt="Sweep configuration" >}}

## 3. agent のローンチ

次に、ローカルで agent をローンチします。作業を分散してより早く sweep ジョブを完了したい場合、最大20台まで異なるマシンで並列に agent をローンチできます。agent は次に試すパラメータのセットを表示します。

{{< img src="/images/sweeps/sweep3.png" alt="Launch agents" >}}

これで sweep が実行中になります。下の画像は例として sweep ジョブが実行されている際のダッシュボードの様子です。[Project ページを見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

{{< img src="/images/sweeps/sweep4.png" alt="Sweep dashboard" >}}

## 既存の run から新しい sweep を seed する

これまでにログした既存の run を使って新しい sweep を開始できます。

1. Project テーブルを開きます。
2. テーブル左側のチェックボックスで使いたい run を選択します。
3. ドロップダウンメニューから新しい sweep の作成をクリックします。

これで sweep がサーバー上にセットされます。あとは一つ以上の agent をローンチして run の実行をスタートするだけです。

{{< img src="/images/sweeps/tutorial_sweep_runs.png" alt="Seed sweep from runs" >}}

{{% alert %}}
新しい sweep をベイズ型 sweep として開始した場合、選択した run もガウス過程の seed として利用されます。
{{% /alert %}}