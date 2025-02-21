---
title: Log objects and media
description: メトリクス 、動画、カスタムプロットなどを追跡する
cascade:
- url: guides/track/log/:filename
menu:
  default:
    identifier: ja-guides-models-track-log-_index
    parent: experiments
url: guides/track/log
weight: 6
---

W&B Python SDKで、メトリクス、メディア、またはカスタム オブジェクトの辞書をステップに記録します。W&B は、各ステップ中にキーと値のペアを収集し、`wandb.log()` でデータを記録するたびに、それらを 1 つの統一された辞書に保存します。スクリプトから記録されたデータは、`wandb` というディレクトリー 内のマシンにローカルに保存され、W&B クラウドまたは [プライベートサーバー]({{< relref path="/guides/hosting/" lang="ja" >}}) に同期されます。

{{% alert %}}
キーと値のペアは、各ステップで同じ値を渡した場合にのみ、1 つの統一された辞書に保存されます。`step` に異なる値を記録すると、W&B は収集されたすべてのキーと値をメモリーに書き込みます。
{{% /alert %}}

`wandb.log` の各呼び出しは、デフォルトで新しい `step` となります。W&B は、チャートと パネル を作成する際に、ステップをデフォルトの x 軸として使用します。オプションで、カスタム x 軸を作成して使用したり、カスタムの概要 メトリクス をキャプチャしたりできます。詳細については、[ログ軸のカスタマイズ]({{< relref path="./customize-logging-axes.md" lang="ja" >}})を参照してください。

{{% alert color="secondary" %}}
`wandb.log()` を使用して、各 `step` の連続した値 (0、1、2 など) を記録します。特定の履歴ステップに書き込むことはできません。W&B は「現在」および「次」のステップにのみ書き込みます。
{{% /alert %}}

## 自動的に記録されるデータ

W&B は、W&B の Experiment 中に次の情報を自動的に記録します。

*   **システムメトリクス**: CPU および GPU の使用率、ネットワークなど。これらは、[run ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}})の [システム] タブに表示されます。GPU の場合、これらは[`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) で取得されます。
*   **コマンドライン**: stdout および stderr が取得され、[run ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}})のログタブに表示されます。

アカウントの [Settings ページ](https://wandb.ai/settings)で [Code Saving](http://wandb.me/code-save-colab) をオンにして、以下を記録します。

*   **Git コミット**: 最新の git コミットを取得し、run ページの Overview タブに表示します。未コミットの変更がある場合は、`diff.patch` ファイルも表示されます。
*   **依存関係**: `requirements.txt` ファイルがアップロードされ、run の `wandb` ディレクトリー に保存したファイルとともに、run ページのファイルタブに表示されます。

## 特定の W&B API 呼び出しで記録されるデータ

W&B を使用すると、記録する内容を正確に決定できます。以下に、一般的に記録されるオブジェクトをいくつか示します。

*   **データセット**: W&B にストリーミングするには、画像またはその他のデータセットサンプルを明示的に記録する必要があります。
*   **プロット**: `wandb.plot` を `wandb.log` とともに使用して、チャートを追跡します。詳細については、[プロットのログ]({{< relref path="./plots.md" lang="ja" >}})を参照してください。
*   **テーブル**: `wandb.Table` を使用してデータを記録し、W&B で視覚化およびクエリを実行します。詳細については、[テーブルのログ]({{< relref path="./log-tables.md" lang="ja" >}})を参照してください。
*   **PyTorch 勾配**: `wandb.watch(model)` を追加して、UI で重みの勾配をヒストグラムとして確認します。
*   **設定 情報**: ハイパーパラメータ、データセットへのリンク、または使用しているアーキテクチャーの名前を設定 パラメータ として記録します。これは、`wandb.init(config=your_config_dictionary)` のように渡されます。詳細については、[PyTorch インテグレーション]({{< relref path="/guides/integrations/pytorch.md" lang="ja" >}})ページを参照してください。
*   **メトリクス**: `wandb.log` を使用して、 model からの メトリクス を確認します。トレーニング ループ 内から精度や損失などの メトリクス を記録すると、UI でライブ更新グラフが表示されます。

## 一般的なワークフロー

1.  **最高精度を比較する**: run 間で メトリクス の最高値を比較するには、その メトリクス の概要値を設定します。デフォルトでは、概要は各キーに記録した最後の値に設定されます。これは UI のテーブルで役立ちます。ここでは、概要 メトリクス に基づいて run をソートおよびフィルタリングし、最終精度ではなく _最高の_ 精度に基づいてテーブルまたは棒グラフで run を比較できます。例: `wandb.run.summary["best_accuracy"] = best_accuracy`
2.  **1 つのチャートに複数の メトリクス を表示する**: `wandb.log({"acc'": 0.9, "loss": 0.1})` のように、複数の メトリクス を同じ `wandb.log` の呼び出しで記録すると、UI で両方をプロットに使用できます
3.  **カスタム x 軸**: 同じログ呼び出しにカスタム x 軸を追加して、W&B ダッシュボード で メトリクス を別の軸に対して視覚化します。例: `wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`。特定の メトリクス のデフォルトの x 軸を設定するには、[Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ja" >}}) を使用します
4.  **リッチメディアとチャートを記録する**: `wandb.log` は、[画像やビデオなどのメディア]({{< relref path="./media.md" lang="ja" >}})から [テーブル]({{< relref path="./log-tables.md" lang="ja" >}})、[チャート]({{< relref path="/guides/models/app/features/custom-charts/" lang="ja" >}})まで、さまざまなデータ型のログ記録をサポートしています。
