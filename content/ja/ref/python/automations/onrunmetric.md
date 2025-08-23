---
title: 'OnRunMetric

  '
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-onrunmetric
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



run のメトリクスがユーザー定義の条件を満たした場合に発生します。

属性:
- event_type (Literal): 説明はありません。
- filter (RunMetricFilter): このイベントでオートメーションをトリガーするために満たす必要がある run やメトリクスの条件。
- scope (ProjectScope): イベントのスコープ。このイベントでは Projects のみ有効です。

### <kbd>メソッド</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントが指定したアクションをトリガーする新しいオートメーションを定義します。
