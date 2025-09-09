---
title: OnCreateArtifact
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-oncreateartifact
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



新しいアーティファクトが作成されます。

属性:
- event_type (Literal): 説明はありません。
- filter (Union): このイベントがオートメーションをトリガーするために、必要に応じて満たす必要がある追加条件。
- scope (Union): イベントのスコープ。このイベントでは、アーティファクト コレクションのみが有効なスコープです。

### <kbd>メソッド</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントが指定されたアクションをトリガーする新しいオートメーションを定義します。