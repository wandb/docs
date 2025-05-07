---
title: ログ オブジェクト と メディア
description: メトリクス、ビデオ、カスタムプロットなどを追跡する
cascade:
- url: /ja/guides/track/log/:filename
menu:
  default:
    identifier: ja-guides-models-track-log-_index
    parent: experiments
url: /ja/guides/track/log
weight: 6
---

W&B Python SDK を使用して、メトリクス、メディア、またはカスタムオブジェクトの辞書をステップにログします。W&B は各ステップごとにキーと値のペアを収集し、`wandb.log()` でデータをログするたびにそれらを統一された辞書に格納します。スクリプトからログされたデータは、`wandb` と呼ばれるディレクトリにローカルに保存され、その後 W&B クラウドまたは [プライベートサーバー]({{< relref path="/guides/hosting/" lang="ja" >}}) に同期されます。

{{% alert %}}
キーと値のペアは、各ステップに同じ値を渡した場合にのみ統一された辞書に保存されます。`step` に異なる値をログした場合、W&B はすべての収集されたキーと値をメモリに書き込みます。
{{% /alert %}}

デフォルトでは、`wandb.log` を呼び出すたびに新しい `step` になります。W&B は、チャートやパネルを作成する際にステップをデフォルトの x 軸として使用します。カスタムの x 軸を作成して使用するか、カスタムの要約メトリックをキャプチャすることも選択できます。詳細は、[ログの軸をカスタマイズする]({{< relref path="./customize-logging-axes.md" lang="ja" >}})を参照してください。

{{% alert color="secondary" %}}
`wandb.log()` を使用して、各 `step` の連続する値をログします: 0, 1, 2, といった具合です。特定の履歴ステップに書き込むことは不可能です。W&B は「現在」と「次」のステップにのみ書き込みます。
{{% /alert %}}

## 自動でログされるデータ

W&B は、W&B Experiment 中に次の情報を自動でログします：

* **システムメトリクス**: CPU と GPU の使用率、ネットワークなど。これらは [run ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) のシステムタブに表示されます。GPU に関しては、[`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) で取得されます。
* **コマンドライン**: stdout と stderr が取得され、[run ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) のログタブに表示されます。

アカウントの [Settings ページ](https://wandb.ai/settings)で[コードの保存](http://wandb.me/code-save-colab)をオンにして、以下をログします：

* **Git コミット**: 最新の git コミットを取得し、run ページの overview タブに表示されます。コミットされていない変更がある場合は `diff.patch` ファイルも表示されます。
* **依存関係**: `requirements.txt` ファイルがアップロードされ、run ページのファイルタブに表示されます。run 用に `wandb` ディレクトリに保存したファイルも含まれます。

## 特定の W&B API 呼び出しでログされるデータ

W&B を使用することで、ログしたいものを正確に決定できます。次に、よくログされるオブジェクトのリストを示します：

* **Datasets**: 画像や他のデータセットサンプルを W&B にストリームするためには、特にログする必要があります。
* **Plots**: グラフを追跡するために `wandb.plot` を `wandb.log` と一緒に使用します。詳細は[ログでのグラフ]({{< relref path="./plots.md" lang="ja" >}})を参照してください。
* **Tables**: `wandb.Table` を使用してデータをログし、W&B でビジュアライズおよびクエリを行います。詳細は[ログでのテーブル]({{< relref path="./log-tables.md" lang="ja" >}})を参照してください。
* **PyTorch 勾配**: モデルの重みの勾配を UI にヒストグラムとして表示するために `wandb.watch(model)` を追加します。
* **設定情報**: ハイパーパラメーター、データセットへのリンク、使用しているアーキテクチャーの名前などを設定パラメーターとしてログします。このように渡します：`wandb.init(config=your_config_dictionary)`。詳細は[PyTorch インテグレーション]({{< relref path="/guides/integrations/pytorch.md" lang="ja" >}})ページをご覧ください。
* **メトリクス**: `wandb.log` を使用してモデルのメトリクスを表示します。トレーニングループ内で精度や損失のようなメトリクスをログすると、UI にライブ更新グラフが表示されます。

## 一般的なワークフロー

1. **最高の精度を比較する**: Runs 間でメトリクスの最高値を比較するには、そのメトリクスの要約値を設定します。デフォルトでは、各キーの最後にログした値が要約に設定されます。これは UI のテーブルで、要約メトリクスに基づいて run を並べ替えたりフィルタリングしたりするのに便利です。_best_ の精度に基づいてテーブルまたは棒グラフで run を比較するのに役立ちます。例：`wandb.run.summary["best_accuracy"] = best_accuracy`
2. **複数のメトリクスを1つのチャートで表示**: `wandb.log` の同じ呼び出し内で複数のメトリクスをログすると、例えばこうなります: `wandb.log({"acc": 0.9, "loss": 0.1})`。UI ではどちらもプロットすることができます。
3. **x 軸をカスタマイズする**: 同じログ呼び出しにカスタム x 軸を追加して、W&B ダッシュボードで別の軸に対してメトリクスを視覚化します。例：`wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`。特定のメトリクスに対するデフォルトの x 軸を設定するには、[Run.define_metric()]({{< relref path="/ref/python/run.md#define_metric" lang="ja" >}}) を使用してください。
4. **リッチメディアとチャートをログする**: `wandb.log` は、[画像やビデオのようなメディア]({{< relref path="./media.md" lang="ja" >}})から[tables]({{< relref path="./log-tables.md" lang="ja" >}})や[charts]({{< relref path="/guides/models/app/features/custom-charts/" lang="ja" >}})に至るまで、多様なデータタイプのログをサポートしています。

## ベストプラクティスとヒント

Experiments やログのためのベストプラクティスとヒントについては、[Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging) を参照してください。