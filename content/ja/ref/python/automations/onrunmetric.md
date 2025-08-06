---
title: OnRunMetric
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}

run メトリックは、ユーザー定義の条件を満たします。

属性:
- event_type (Literal): 説明はありません。
- filter (RunMetricFilter): このイベントがオートメーションをトリガーするために満たす必要がある Run やメトリックの条件。
- scope (ProjectScope): イベントのスコープ。このイベントで有効なのは Project のみです。

### <kbd>メソッド</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントが指定されたアクションをトリガーする新しいオートメーションを定義します。
