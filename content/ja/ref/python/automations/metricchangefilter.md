---
title: 'MetricChangeFilter

  '
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}



ユーザーが定義したしきい値に対して、run のメトリクスの変化を比較するフィルターを定義します。

この変化は「タンブリング」ウィンドウ、すなわち現在のウィンドウと重複しない直前のウィンドウとの差分として計算されます。

属性:
- agg (オプション): ウィンドウサイズに対して適用する集約（aggregate）操作。
- change_dir (ChangeDir): 説明はありません。
- change_type (ChangeType): 説明はありません。
- name (str): 監視するメトリクスの名前。
- prior_window (int): メトリクスを集約する直前ウィンドウのサイズ（`agg is None` の場合は無視されます）。
    省略時は、現在のウィンドウサイズが既定値となります。
- threshold (Union): 比較対象となるしきい値。
- window (int): メトリクスを集約するウィンドウのサイズ（`agg is None` の場合は無視されます）。