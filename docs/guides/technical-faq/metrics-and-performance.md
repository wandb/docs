---
displayed_sidebar: default
---


# Metrics & Performance

## Metrics

### システムメトリクスはどのくらいの頻度で収集されますか？

デフォルトでは、メトリクスは2秒ごとに収集され、15秒間隔で平均化されます。より高解像度のメトリクスが必要な場合は、[contact@wandb.com](mailto:contact@wandb.com) までご連絡ください。

### メトリクスだけをログに記録できますか？コードやデータセットの例は不要ですか？

**Dataset Examples**

デフォルトでは、データセットの例は記録されません。この機能をオンにすると、ウェブインターフェースで予測例を確認できます。

**Code Logging**

コードログの記録をオフにする方法は2つあります：

1. `WANDB_DISABLE_CODE` を `true` に設定して、すべてのコード追跡をオフにします。この場合、git SHAや差分パッチは記録されません。
2. `WANDB_IGNORE_GLOBS` を `*.patch` に設定して、差分パッチのサーバーへの同期をオフにします。ローカルには保存され、`wandb restore` で適用可能です。

### 異なる時間スケールでメトリクスを記録できますか？（例: 各バッチごとのトレーニング精度とエポックごとの検証精度を記録したい）

はい、可能です。メトリクスを記録する際にインデックス（例: `batch` や `epoch`）も一緒にログに記録すればよいです。一つのステップで `wandb.log({'train_accuracy': 0.9, 'batch': 200})` を記録し、別のステップで `wandb.log({'val_accuracy': 0.8, 'epoch': 4})` を記録します。その後、UIで各チャートのx軸に適切な値を設定できます。特定のインデックスのデフォルトx軸を設定したい場合は、[Run.define\_metric()](../../ref/python/run.md#define_metric) を使用します。上記の例では次のようになります：

```python
wandb.init()

wandb.define_metric("batch")
wandb.define_metric("epoch")

wandb.define_metric("train_accuracy", step_metric="batch")
wandb.define_metric("val_accuracy", step_metric="epoch")
```

### 時間とともに変化しないメトリクス（例: 最終評価精度）を記録する方法は？

`wandb.log({'final_accuracy': 0.9})` を使用すれば十分です。デフォルトでは `wandb.log({'final_accuracy'})` は `wandb.settings['final_accuracy']` を更新し、その値がRunsテーブルに表示されます。

### runが完了した後に追加のメトリクスを記録するにはどうすればよいですか？

いくつかの方法があります。

複雑なワークフローの場合、複数のRunsを使用し、[`wandb.init`](../track/launch.md) でgroupパラメーターを使い、単一の実験の一部としてrunするすべてのプロセス内で一意の値を設定することをお勧めします。[runs table](../app/pages/run-page.md) はグループIDで自動的にテーブルをグループ化し、可視化が期待通りに動作します。これにより、複数のExperimentsやトレーニングRunsを別々のプロセスとして実行し、すべての結果を一つの場所に記録できます。

簡単なワークフローの場合は、`resume=True` および `id=UNIQUE_ID` を設定して `wandb.init` を呼び出し、その後同じ `id=UNIQUE_ID` で `wandb.init` を再度呼び出します。その後、[`wandb.log`](../track/log/intro.md) または `wandb.summary` で通常通りログを記録し、Runsの値を更新できます。

## Performance

### wandbはトレーニングのパフォーマンスを低下させますか？

通常の使用であれば、W&Bはトレーニングのパフォーマンスにほとんど影響を与えません。通常の使用とは、1秒あたり1回未満のログと、各ステップで数メガバイト以下のデータを記録することを意味します。W&Bは別のプロセスで実行され、関数呼び出しはブロックされないため、ネットワークが一時的にダウンした場合やディスクの読み書きに断続的な問題が生じた場合でも、パフォーマンスには影響しません。ただし、大量のデータを急速に記録する場合、ディスクI/O問題が発生する可能性があります。ご質問があれば、お気軽にご連絡ください。

### プロジェクトごとに作成するべきrunの数は？

パフォーマンス上の理由から、1プロジェクトあたり約10k runを推奨します。

### ハイパーパラメーター検索のベストプラクティス

1プロジェクトあたり10k run（約）は合理的な限界であり、その場合の推奨事項は、`wandb.init()` でタグを設定し、各検索に一意のタグを持たせることです。これにより、ProjectsページのRunsテーブルでそのタグをクリックすることで、簡単に特定の検索にフィルタリングできます。例えば `wandb.init(tags='your_tag')` docsは[こちら](../../ref/python/init.md)で確認できます。