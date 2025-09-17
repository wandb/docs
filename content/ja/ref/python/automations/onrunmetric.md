---
title: OnRunMetric
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-onrunmetric
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



Run のメトリクスがユーザー定義の条件を満たしたときのイベント。

属性:
- event_type (Literal): 説明はありません。
- filter (RunMetricFilter): このイベントが オートメーション をトリガーするために満たす必要がある Run および/または メトリクス の条件。
- scope (ProjectScope): イベントのスコープ。このイベントでは プロジェクト のみが有効なスコープです。

### <kbd>メソッド</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントが指定したアクションをトリガーする新しい オートメーション を定義します。