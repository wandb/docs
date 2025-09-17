---
title: オブジェクトとメディアをログする
description: メトリクス、動画、カスタムプロットなどを追跡
cascade:
- url: guides/track/log/:filename
menu:
  default:
    identifier: ja-guides-models-track-log-_index
    parent: experiments
url: guides/track/log
weight: 6
---

W&B Python SDK で、メトリクス、メディア、またはカスタム オブジェクトの辞書を step にログします。W&B は各 step ごとにキーと値のペアを収集し、`wandb.Run.log()` でデータをログするたびにそれらを 1 つの統合された辞書に保存します。スクリプトからログされたデータは、まずローカルマシンの `wandb` というディレクトリーに保存され、その後 W&B のクラウドまたはあなたの [private server]({{< relref path="/guides/hosting/" lang="ja" >}}) に同期されます。

{{% alert %}}
キーと値のペアは、各 step に同じ 値 を渡した場合にのみ、1 つの統合された辞書に格納されます。`step` に異なる 値 をログした場合、W&B は収集したキーと値をすべてメモリに書き込みます。
{{% /alert %}}

`wandb.Run.log()` を呼ぶたびに、デフォルトでは新しい `step` になります。W&B はチャートやパネルを作成する際のデフォルトの x 軸として step を使用します。必要に応じてカスタムの x 軸を作成して使ったり、カスタムのサマリー メトリクスを記録したりできます。詳しくは [Customize log axes]({{< relref path="./customize-logging-axes.md" lang="ja" >}}) を参照してください。









{{% alert color="secondary" %}}
`wandb.Run.log()` を使って、各 `step` に対して 0、1、2、… のように連続する値をログしてください。特定の履歴の step に書き込むことはできません。W&B は "current" と "next" の step にのみ書き込みます。
{{% /alert %}}





## Automatically logged data

W&B は W&B Experiment の実行中に、以下の情報を自動でログします:


* **System metrics**: CPU や GPU の使用率、ネットワークなど。GPU については、[`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) で取得します。
* **Command line**: stdout と stderr を取得し、[run page]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の logs タブに表示します。

アカウントの [Settings ページ](https://wandb.ai/settings) で [Code Saving](https://wandb.me/code-save-colab) をオンにすると、次もログされます:

* **Git commit**: 最新の git commit を取得し、run ページの Overview タブに表示します。未コミットの変更がある場合は `diff.patch` ファイルも表示されます。
* **Dependencies**: `requirements.txt` がアップロードされ、run ページの Files タブに表示されます。加えて、その run のために `wandb` ディレクトリーに保存したファイルも表示されます。


## What data is logged with specific W&B API calls?

W&B では、何をログするかを正確にコントロールできます。以下はよくログされるオブジェクトの例です:

* **Datasets**: 画像やその他のデータセット サンプルを W&B にストリームしたい場合は、明示的にログする必要があります。
* **Plots**: `wandb.plot()` と `wandb.Run.log()` を組み合わせてチャートを追跡します。詳しくは [プロットをログする]({{< relref path="./plots.md" lang="ja" >}}) を参照してください。 
* **Tables**: `wandb.Table` を使って、W&B で可視化・クエリできるデータをログします。詳しくは [テーブルをログする]({{< relref path="./log-tables.md" lang="ja" >}}) を参照してください。
* **PyTorch の勾配**: `wandb.Run.watch(model)` を追加すると、重みの勾配を UI でヒストグラムとして確認できます。
* **Configuration information**: ハイパーパラメーター、データセットへのリンク、使用しているアーキテクチャー名などを config パラメータとしてログします。渡し方の例: `wandb.init(config=your_config_dictionary)`。詳しくは [PyTorch インテグレーション]({{< relref path="/guides/integrations/pytorch.md" lang="ja" >}}) を参照してください。 
* **Metrics**: `wandb.Run.log()` を使ってモデルのメトリクスを可視化します。トレーニング ループ内から精度や損失などをログすると、UI 上でグラフがライブ更新されます。



## Common workflows

1. **最良の精度を比較する**: run をまたいであるメトリクスの最良の値を比較するには、そのメトリクスのサマリー 値を設定します。デフォルトでは、サマリーは各キーに対して最後にログした値になります。これは UI のテーブルで便利で、サマリー メトリクスに基づいて run をソート・フィルタでき、最終精度ではなく _最良_ の精度に基づいて、テーブルや棒グラフで run を比較できます。例: `wandb.run.summary["best_accuracy"] = best_accuracy`
2. **1 つのチャートに複数のメトリクスを表示する**: 次のように同じ `wandb.Run.log()` 呼び出しで複数のメトリクスをログします: `wandb.log({"acc'": 0.9, "loss": 0.1})`。これらはどちらも UI でプロットに使えるようになります。
3. **x 軸をカスタマイズする**: 同じログ呼び出しにカスタムの x 軸を追加して、W&B ダッシュボードで別の軸に対してメトリクスを可視化します。例: `wandb.Run.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`。特定のメトリクスのデフォルト x 軸を設定するには [Run.define_metric()]({{< relref path="/ref/python/sdk/classes/run.md#define_metric" lang="ja" >}}) を使用します。
4. **リッチなメディアやチャートをログする**: `wandb.Run.log()` は、[画像や動画などのメディア]({{< relref path="./media.md" lang="ja" >}}) から [テーブル]({{< relref path="./log-tables.md" lang="ja" >}})、[チャート]({{< relref path="/guides/models/app/features/custom-charts/" lang="ja" >}}) まで、幅広いデータ型のログをサポートしています。

## Best practices and tips 

Experiments とログのベストプラクティスとヒントは、[ベストプラクティス: Experiments と ログ](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging) を参照してください。