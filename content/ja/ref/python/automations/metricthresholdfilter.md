---
title: 'MetricThresholdFilter

  '
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}



run のメトリクスをユーザーが定義したしきい値と比較するフィルターを定義します。

属性:
- agg（オプション）: ウィンドウサイズに対して適用される集約処理（省略可）。
- cmp（リテラル）: メトリクスの値（左側）としきい値（右側）を比較する際に使用される比較演算子。
- name（str）: 監視するメトリクス名。
- threshold（Union）: 比較対象となるしきい値。
- window（int）: メトリクスが集約されるウィンドウのサイズ（`agg` が None の場合は無視されます）。