---
title: 'OnCreateArtifact

  '
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-oncreateartifact
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



新しい artifact が作成されます。

属性:
- event_type (Literal): 説明はありません。
- filter (Union): このイベントがオートメーションをトリガーするために、追加で満たす必要がある条件（もしあれば）。
- scope (Union): イベントのスコープ。このイベントでは artifact コレクションのみが有効なスコープです。

### <kbd>method</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントによって指定されたアクションがトリガーされる新しいオートメーションを定義します。
