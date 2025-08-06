---
title: オブジェクトとメディアのログ
description: メトリクス、動画、カスタムプロットなどを記録して管理しましょう
cascade:
- url: guides/track/log/:filename
menu:
  default:
    identifier: ja-guides-models-track-log-_index
    parent: experiments
url: guides/track/log
weight: 6
---

W&B Python SDK を使って、メトリクス、メディア、カスタムオブジェクトの辞書をステップごとにログできます。W&B は各ステップでキーと値のペアを収集し、`wandb.Run.log()` でデータを記録するたびに、これらをひとつの統一された辞書として保存します。スクリプトからログしたデータは、ローカルの `wandb` ディレクトリーに保存され、その後 W&B クラウドや [プライベートサーバー]({{< relref path="/guides/hosting/" lang="ja" >}}) に同期されます。

{{% alert %}}
キーと値のペアは、各ステップで同じ値を渡した場合にのみ、ひとつの統一された辞書として保存されます。もし `step` に異なる値をログする場合、W&B はすべてのキーと値をメモリに書き込みます。
{{% /alert %}}

`wandb.Run.log()` を呼び出すごとに、デフォルトで新しい `step` になります。W&B は、チャートやパネルを作成する際、デフォルトの x 軸としてステップを利用します。必要に応じてカスタムの x 軸や、カスタムサマリメトリクスを記録することも可能です。 詳しくは [ログ軸のカスタマイズ]({{< relref path="./customize-logging-axes.md" lang="ja" >}})をご覧ください。

{{% alert color="secondary" %}}
`wandb.Run.log()` を使えば、`step` ごとに連続する値（0, 1, 2 ...）をログできます。特定の履歴ステップに書き込むことはできません。W&B が書き込むのは「現在」または「次」のステップだけです。
{{% /alert %}}

## 自動ログされるデータ

W&B は W&B Experiment 中に以下の情報を自動でログします：

* **システムメトリクス**：CPU/GPU の使用率、ネットワークなど。GPU に関しては、[`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) から取得されます。
* **コマンドライン**：stdout や stderr が取得され、[run 実行ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}})の logs タブに表示されます。

アカウントの [設定ページ](https://wandb.ai/settings) で [コード保存](https://wandb.me/code-save-colab) をオンにすると、以下もログされます：

* **Git コミット**：最新の git コミットが pick up され、run ページの Overviewタブ に表示されます。未コミットの変更がある場合は `diff.patch` ファイルも付きます。
* **依存関係**：`requirements.txt` ファイルがアップロードされ、run ページの files タブに表示されます。このほか、run のために `wandb` ディレクトリーへ保存したファイルも同様に表示されます。

## 特定の W&B API コールでログされるデータ

W&B では、ログする内容を柔軟に選択できます。よく使われるオブジェクトの例は次の通りです：

* **Datasets**：画像やその他のサンプルを明示的にログすれば、W&B にストリーム配信できます。
* **プロット**：`wandb.plot()` と `wandb.Run.log()` を組み合わせてチャートを記録しましょう。詳細は [プロットのログ]({{< relref path="./plots.md" lang="ja" >}}) を参照してください。
* **Tables**：`wandb.Table` を使ってデータをログすれば、W&B 上でビジュアライズやクエリが可能です。[Tables の記録]({{< relref path="./log-tables.md" lang="ja" >}}) の詳細もご覧ください。
* **PyTorch の勾配**：`wandb.Run.watch(model)` を追加すると、重みの勾配を UI 上にヒストグラムとして表示できます。
* **設定情報**：ハイパーパラメーターやデータセットへのリンク、利用しているアーキテクチャー名などを設定パラメータとして記録できます。例：`wandb.init(config=your_config_dictionary)`。 詳細は [PyTorch インテグレーション]({{< relref path="/guides/integrations/pytorch.md" lang="ja" >}}) をご覧ください。
* **メトリクス**：`wandb.Run.log()` でモデルのメトリクスを記録しましょう。トレーニングループ内で accuracy や loss などをログすると、UI 上でライブ更新グラフが得られます。

## よく使われるワークフロー

1. **ベスト accuracy の比較**：メトリクスのベスト値を run 間で比較したい時には、summary 値をセットします。デフォルトでは、summary は各キーで最後にログした値になります。これは、UI 内のテーブルで run を summary メトリクスでソート・フィルタし、_ベスト_ accuracy で比較できるため便利です。例：`wandb.run.summary["best_accuracy"] = best_accuracy`
2. **複数メトリクスを1つのグラフに**：`wandb.Run.log()` に複数のメトリクスをまとめて渡します。例：`wandb.log({"acc": 0.9, "loss": 0.1})` こうすると、両方の値をプロットで比較できます。
3. **x軸のカスタマイズ**：同じログコールでカスタム x 軸を含めると、W&B ダッシュボードで他の軸を使ってメトリクスを可視化できます。例：`wandb.Run.log({'acc': 0.9, 'epoch': 3, 'batch': 117})`。特定メトリクスのデフォルト x軸を指定するなら [Run.define_metric()]({{< relref path="/ref/python/sdk/classes/run.md#define_metric" lang="ja" >}}) をご利用ください。
4. **リッチメディアやチャートのログ**：`wandb.Run.log()` は、[画像・動画などのメディア]({{< relref path="./media.md" lang="ja" >}})、[Tables]({{< relref path="./log-tables.md" lang="ja" >}})、[チャート]({{< relref path="/guides/models/app/features/custom-charts/" lang="ja" >}}) など、さまざまなデータタイプの記録に対応しています。

## ベストプラクティスとヒント

Experiments やログ手法のベストプラクティスやヒントは [Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging) をご覧ください。