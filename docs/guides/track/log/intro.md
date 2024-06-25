---
description: メトリクス、ビデオ、カスタムプロットなどを追跡する
slug: /guides/track/log
displayed_sidebar: default
---


# Log Media and Objects in Experiments

<head>
  <title>Log Media and Objects in Experiments</title>
</head>

辞書形式のメトリクス、メディア、またはカスタムオブジェクトを W&B Python SDK を使ってステップごとにログします。W&B は各ステップでキーと値のペアを収集し、`wandb.log()` を使ってデータをログするたびにそれらを統合された一つの辞書に保存します。あなたのスクリプトからログされたデータはローカルのマシンに `wandb` というディレクトリーに保存され、その後 W&B クラウドまたはあなたの [プライベートサーバー](../../hosting/intro.md) に同期されます。

:::info
キーと値のペアは各ステップで同じ値を渡した場合のみ統合された一つの辞書に保存されます。異なる値を `step` にログすると、W&B は収集されたすべてのキーと値をメモリーに書き込みます。
:::

デフォルトでは、`wandb.log` の各呼び出しは新しい `step` です。W&B はチャートやパネルを作成する際にステップをデフォルトの x 軸として使用します。カスタム x 軸を作成したり、カスタムサマリーメトリクスをキャプチャしたりすることもできます。詳細については、[ログ軸のカスタマイズ](./customize-logging-axes.md)を参照してください。

:::caution
各 `step` に対して連続する値: 0, 1, 2, を `wandb.log()` を使ってログしてください。特定の履歴ステップに書き込むことはできません。W&B は常に "current" と "next" のステップにのみ書き込みます。
:::

## 自動的にログされるデータ

W&Bは以下の情報を W&B Experiment 中に自動的にログします:

* **システムメトリクス**: CPU や GPU の使用率、ネットワークなど。これらは [run ページ](../../app/pages/run-page.md) の System タブに表示されます。GPU に関しては、[`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) を使って取得されます。
* **コマンドライン**: stdout および stderr が取得され、[run ページ](../../app/pages/run-page.md) のログタブに表示されます。

アカウントの [Settings ページ](https://wandb.ai/settings) で [Code Saving](http://wandb.me/code-save-colab) をオンにすると以下もログされます:

* **Git コミット**: 最新の git コミットを取得し、run ページの Overviewタブで確認できます。また、未コミットの変更がある場合は `diff.patch` ファイルも表示されます。
* **依存関係**: `requirements.txt` ファイルがアップロードされ、run ページのファイルタブに表示されます。さらに、`wandb` ディレクトリーに保存するファイルも同様に表示されます。

## 特定の W&B API 呼び出しでログされるデータは？

W&B では、ログする内容を正確に決定することができます。以下はいくつかの一般的なログ対象のオブジェクトです:

* **Datasets**: 画像や他のデータセットサンプルを特にログする必要があります。
* **Plots**: `wandb.plot` と `wandb.log` を使ってチャートを追跡します。詳細は [Log Plots](./plots.md) を参照してください。
* **Tables**: `wandb.Table` を使ってデータをログし、W&Bで視覚化およびクエリできます。詳細は [Log Tables](./log-tables.md) を参照してください。
* **PyTorch gradients**: `wandb.watch(model)` を追加して、UI でウエイトの勾配をヒストグラムとして表示します。
* **設定情報**: ハイパーパラメーター、データセットへのリンク、使用しているアーキテクチャーの名前などを設定パラメーターとしてログします。このように渡します: `wandb.init(config=your_config_dictionary)`。詳細は [PyTorch Integrations](../../integrations/pytorch.md) ページを参照してください。
* **メトリクス**: `wandb.log` を使ってモデルのメトリクスを表示します。トレーニングループの中から精度や損失のようなメトリクスをログすると、UI上でリアルタイムで更新されるグラフが得られます。

## 一般的なワークフロー

1. **最高の精度を比較する**: 複数の run 間でメトリクスの最高値を比較するには、そのメトリクスのサマリー値を設定します。デフォルトでは、サマリーは各キーに対してログした最後の値に設定されます。これは UI のテーブルで役立ち、サマリーメトリクスに基づいて run をソートおよびフィルターできます。たとえば、最終精度ではなく最高精度に基づいて run をテーブルやバーチャートで比較できます。サマリーを設定する例は次の通りです: `wandb.run.summary["best_accuracy"] = best_accuracy`
2. **複数のメトリクスを一つのチャートに**: 同じ `wandb.log` 呼び出しで複数のメトリクスをログする例: `wandb.log({"acc'": 0.9, "loss": 0.1})`。これにより、UI 上で両方をプロットに使用できます。
3. **カスタム x 軸**: カスタム x 軸を同じログ呼び出しに追加して、W&B ダッシュボードで異なる軸に対してメトリクスを視覚化します。例: `wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`。特定のメトリクスにデフォルトの x 軸を設定するには、[Run.define_metric()](../../../ref/python/run.md#define_metric) を使用します。
4. **リッチメディアとチャートをログする**: `wandb.log` は画像やビデオのような [メディア](./media.md) から、[テーブル](./log-tables.md) や [チャート](../../app/features/custom-charts/intro.md) まで、多様なデータ型のログをサポートします。