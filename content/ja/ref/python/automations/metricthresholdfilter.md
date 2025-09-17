---
title: MetricThresholdFilter
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-metricthresholdfilter
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}



ユーザー定義のしきい値に対して run メトリクスを比較するフィルターを定義します。

属性:
- agg (Optional): ウィンドウのサイズ全体に適用する集約処理（ある場合）。
- cmp (Literal): メトリクスの値（左）としきい値（右）を比較するために使われる比較演算子。
- name (str): 観測対象のメトリクス名。
- threshold (Union): 比較対象のしきい値。
- window (int): メトリクスを集約するウィンドウのサイズ（ `agg is None` の場合は無視されます）。