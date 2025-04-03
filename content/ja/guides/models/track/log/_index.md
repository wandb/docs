---
title: Log objects and media
description: メトリクス 、動画、カスタムプロットなどを追跡
cascade:
- url: guides/track/log/:filename
menu:
  default:
    identifier: ja-guides-models-track-log-_index
    parent: experiments
url: guides/track/log
weight: 6
---

W&B Python SDK を使用して、メトリクス、メディア、またはカスタム オブジェクトの辞書をステップと共にログに記録します。W&B は、各ステップ中にキーと値のペアを収集し、`wandb.log()` でデータをログに記録するたびに、それらを 1 つの統合された辞書に保存します。スクリプトからログに記録されたデータは、ローカル マシンの `wandb` というディレクトリーに保存され、W&B クラウドまたは [プライベート サーバー]({{< relref path="/guides/hosting/" lang="ja" >}}) に同期されます。

{{% alert %}}
キーと値のペアは、各ステップで同じ値を渡す場合にのみ、1 つの統合された辞書に保存されます。`step` に対して異なる値をログに記録すると、W&B は収集されたすべてのキーと値をメモリーに書き込みます。
{{% /alert %}}

`wandb.log` の各呼び出しは、デフォルトで新しい `step` となります。W&B は、チャートとパネルを作成する際に、ステップをデフォルトの x 軸として使用します。オプションで、カスタム x 軸を作成して使用したり、カスタムの集計メトリクスをキャプチャしたりできます。詳細については、[ログ軸のカスタマイズ]({{< relref path="./customize-logging-axes.md" lang="ja" >}}) を参照してください。

{{% alert color="secondary" %}}
`wandb.log()` を使用して、各 `step` に対して連続した値 (0、1、2 など) をログに記録します。特定の履歴ステップに書き込むことはできません。W&B は、「現在」および「次」のステップにのみ書き込みます。
{{% /alert %}}

## 自動的にログに記録されるデータ

W&B は、W&B の Experiments 中に次の情報を自動的にログに記録します。

* **システム メトリクス**: CPU と GPU の使用率、ネットワークなど。これらは、[run ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の [System] タブに表示されます。GPU の場合、これらは [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) で取得されます。
* **コマンドライン**: stdout と stderr が取得され、[run ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) の [Logs] タブに表示されます。

アカウントの [Settings page](https://wandb.ai/settings) で [Code Saving](http://wandb.me/code-save-colab) をオンにして、以下をログに記録します。

* **Git commit**: 最新の git commit を取得し、run ページの Overview タブに表示します。また、コミットされていない変更がある場合は、`diff.patch` ファイルも表示します。
* **依存関係**: `requirements.txt` ファイルがアップロードされ、run の files タブに表示されます。また、run の `wandb` ディレクトリーに保存したファイルも表示されます。

## 特定の W&B API 呼び出しでログに記録されるデータ

W&B を使用すると、ログに記録する内容を正確に決定できます。以下に、一般的にログに記録されるオブジェクトをいくつか示します。

* **Datasets**: 画像またはその他の dataset サンプルを W&B にストリーミングするには、それらを具体的にログに記録する必要があります。
* **プロット**: `wandb.plot` を `wandb.log` と共に使用して、チャートを追跡します。詳細については、[プロットのログ]({{< relref path="./plots.md" lang="ja" >}}) を参照してください。
* **Tables**: `wandb.Table` を使用してデータをログに記録し、W&B で視覚化およびクエリを実行します。詳細については、[Tables のログ]({{< relref path="./log-tables.md" lang="ja" >}}) を参照してください。
* **PyTorch 勾配**: `wandb.watch(model)` を追加して、UI で重みの勾配をヒストグラムとして表示します。
* **設定情報**: ハイパーパラメータ、dataset へのリンク、または使用しているアーキテクチャーの名前を config パラメータとしてログに記録します。`wandb.init(config=your_config_dictionary)` のように渡されます。詳細については、[PyTorch Integrations]({{< relref path="/guides/integrations/pytorch.md" lang="ja" >}}) ページを参照してください。
* **メトリクス**: `wandb.log` を使用して、model からのメトリクスを表示します。トレーニング ループ内から精度や損失などのメトリクスをログに記録すると、UI でライブ更新グラフが表示されます。

## 一般的なワークフロー

1. **最高精度を比較する**: run 全体でメトリクスの最高値を比較するには、そのメトリクスの集計値を設定します。デフォルトでは、集計は各キーに対してログに記録した最後の値に設定されます。これは、UI のテーブルで役立ちます。UI では、集計メトリクスに基づいて run をソートおよびフィルター処理し、最終的な精度ではなく _最高_ 精度に基づいてテーブルまたは棒グラフで run を比較できます。例: `wandb.run.summary["best_accuracy"] = best_accuracy`
2. **1 つのチャートで複数のメトリクスを表示する**: `wandb.log({"acc'": 0.9, "loss": 0.1})` のように、`wandb.log` への同じ呼び出しで複数のメトリクスをログに記録すると、それらは両方とも UI でプロットに使用できるようになります。
3. **x 軸をカスタマイズする**: 同じログ呼び出しにカスタム x 軸を追加して、W&B dashboard で別の軸に対してメトリクスを視覚化します。例: `wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`。特定のメトリクスのデフォルトの x 軸を設定するには、[Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ja" >}}) を使用します。
4. **リッチ メディアとチャートをログに記録する**: `wandb.log` は、[画像や動画などのメディア]({{< relref path="./media.md" lang="ja" >}}) から [Tables]({{< relref path="./log-tables.md" lang="ja" >}}) および [Charts]({{< relref path="/guides/models/app/features/custom-charts/" lang="ja" >}}) まで、さまざまなデータ型のログ記録をサポートしています。

## ベストプラクティスとヒント

Experiments とログ記録のベストプラクティスとヒントについては、[ベストプラクティス: Experiments とログ記録](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging) を参照してください。
