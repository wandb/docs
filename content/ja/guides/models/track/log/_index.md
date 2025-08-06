---
title: オブジェクトとメディアをログする
description: メトリクス、動画、カスタムプロットなどを記録して管理しましょう
menu:
  default:
    identifier: log-objects-and-media
    parent: experiments
weight: 6
url: guides/track/log
cascade:
- url: guides/track/log/:filename
---

W&B Python SDK でメトリクス、メディア、カスタムオブジェクトの辞書を step ごとにログできます。 W&B は各 step でキーと値のペアを収集し、`wandb.Run.log()` でデータをログするたびに 1 つの統合辞書として保存します。スクリプトからログされたデータは、ローカルマシンの `wandb` というディレクトリーに保存され、その後 W&B クラウドまたは [プライベートサーバー]({{< relref "/guides/hosting/" >}}) に同期されます。

{{% alert %}}
キーと値のペアは、各 step で同じ値を渡した場合のみ 1 つの統合辞書に保存されます。`step` に異なる値をログすると、W&B は収集したすべてのキーと値をメモリに書き込みます。
{{% /alert %}}

`wandb.Run.log()` への各呼び出しはデフォルトで新しい `step` となります。W&B は step をチャートやパネルを作成する際のデフォルトの x 軸として使用します。カスタム x 軸を作成して利用したり、カスタムサマリーメトリクスを取得することも可能です。詳しくは [ログの軸をカスタマイズ]({{< relref "./customize-logging-axes.md" >}}) をご覧ください。

{{% alert color="secondary" %}}
`wandb.Run.log()` を使い、各 `step`（0, 1, 2, ...）の連続した値をログしてください。特定の履歴 step への書き込みはできません。W&B は「現在」と「次」の step へのみ書き込みを行います。
{{% /alert %}}

## 自動でログされるデータ

W&B は W&B Experiment の実行時に以下の情報を自動でログします。

* **システムメトリクス**: CPU や GPU の使用率、ネットワーク等。GPU の場合は [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) で取得します。
* **コマンドライン**: stdout および stderr の内容が取得され、[run ページ]({{< relref "/guides/models/track/runs/" >}})の logs タブに表示されます。

アカウントの [Settings ページ](https://wandb.ai/settings) で [Code Saving](https://wandb.me/code-save-colab) をオンにすると、以下もログします:

* **Git コミット**: 最新の git コミットを取得し、run ページの Overview タブや、未コミットの変更がある場合は `diff.patch` ファイルとして表示します。
* **依存関係**: `requirements.txt` ファイルをアップロードし、run ページの Files タブで確認できます。他にも run 用に `wandb` ディレクトリーに保存したファイルも表示されます。

## 特定の W&B API コールでログされるデータ

W&B では、何をログするかを柔軟に選択できます。よく使われるログ対象は以下の通りです：

* **Datasets**: 画像やサンプルなど、Datasets を W&B にストリームするには明示的にログが必要です。
* **Plots**: `wandb.plot()` と `wandb.Run.log()` を組み合わせることでチャートのトラッキングが可能です。詳細は [プロットのログ]({{< relref "./plots.md" >}}) をご参照ください。
* **Tables**: `wandb.Table` を使ってデータをログし、W&B 上で可視化やクエリが行えます。詳細は [テーブルのログ]({{< relref "./log-tables.md" >}}) をご覧ください。
* **PyTorch 勾配**: `wandb.Run.watch(model)` を追加すると、UI 上で重みの勾配をヒストグラムとして確認できます。
* **設定情報**: ハイパーパラメーター、データセットへのリンク、利用するアーキテクチャー名などを config パラメータとしてログします。`wandb.init(config=your_config_dictionary)` のように渡してください。詳細は [PyTorch インテグレーション]({{< relref "/guides/integrations/pytorch.md" >}}) ページを参照してください。
* **メトリクス**: `wandb.Run.log()` でモデルからのメトリクスを表示できます。トレーニングループ内で精度や損失などのメトリクスをログすると、UI 上でグラフがリアルタイムに更新されます。

## 一般的なワークフロー

1. **ベストな精度を比較する**: 複数 run でメトリクスのベスト値を比較したい場合、そのメトリクスの summary 値を設定します。デフォルトでは summary は各キーの最後にログした値になります。これにより UI のテーブルで summary メトリクスで run をソート・フィルターし、テーブルやバーチャート上で _ベスト_ な精度を比較できます。例：`wandb.run.summary["best_accuracy"] = best_accuracy`
2. **複数メトリクスを 1 つのチャートで表示**: `wandb.Run.log()` を 1 回呼び出し、複数のメトリクスを同時にログできます。例：`wandb.log({"acc": 0.9, "loss": 0.1})` で、どちらも UI のグラフで選択可能となります。
3. **x 軸をカスタマイズ**: カスタム x 軸も同じログ呼び出しで渡せます。W&B ダッシュボード上で異なる軸とメトリクスを可視化できます。例：`wandb.Run.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`。特定メトリクス用のデフォルト x 軸は [Run.define_metric()]({{< relref "/ref/python/sdk/classes/run.md#define_metric" >}}) で設定できます。
4. **リッチなメディアやチャートのログ**: `wandb.Run.log()` は [画像や動画]({{< relref "./media.md" >}}) などのメディア、[テーブル]({{< relref "./log-tables.md" >}})、[カスタムチャート]({{< relref "/guides/models/app/features/custom-charts/" >}}) など多様なデータタイプに対応しています。

## ベストプラクティスとヒント

実験管理とログにおけるベストプラクティスやヒントは [Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging) をご覧ください。