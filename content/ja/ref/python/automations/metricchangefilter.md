---
title: MetricChangeFilter
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-metricchangefilter
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}



run メトリクスの変化をユーザー定義のしきい値と比較するフィルターを定義します。

この変化は "tumbling" ウィンドウ、すなわち現在のウィンドウと重なりのない直前のウィンドウとの差分として計算されます。

属性:
- agg (Optional): ウィンドウ サイズに対して適用する集約処理（ある場合）。
- change_dir (ChangeDir): 説明はありません。
- change_type (ChangeType): 説明はありません。
- name (str): 監視対象のメトリクス名。
- prior_window (int): メトリクスを集約する直前ウィンドウのサイズ（`agg is None` の場合は無視されます）。
    省略時は現在のウィンドウのサイズがデフォルトになります。
- threshold (Union): 比較対象となるしきい値。
- window (int): メトリクスを集約するウィンドウのサイズ（`agg is None` の場合は無視されます）。