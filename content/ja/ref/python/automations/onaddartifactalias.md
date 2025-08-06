---
title: OnAddArtifactAlias
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}

新しいエイリアスがアーティファクトに割り当てられます。

属性:
- event_type (Literal): 説明はありません。
- filter (Union): このイベントがオートメーションをトリガーするために満たす必要がある、追加の条件（ある場合）。
- scope (Union): このイベントのスコープ。

### <kbd>メソッド</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントが指定されたアクションをトリガーする新しいオートメーションを定義します。
