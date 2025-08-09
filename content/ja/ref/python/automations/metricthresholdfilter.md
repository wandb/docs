---
title: 'MetricThresholdFilter

  '
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-metricthresholdfilter
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}

run のメトリクスをユーザーが設定したしきい値と比較するフィルターを定義します。

属性:
- agg（オプション）: ウィンドウサイズに対して適用する集約処理（指定があれば）。
- cmp（リテラル）: メトリクス値（左辺）としきい値（右辺）を比較する際に使う比較演算子。
- name（str）: 監視対象のメトリクス名。
- threshold（Union）: 比較対象となるしきい値。
- window（int）: メトリクスを集約するウィンドウサイズ（`agg` が None の場合は無視されます）。