---
title: OnCreateArtifact
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



新しい artifact が作成されます。

属性:
- event_type (Literal): 説明はありません。
- filter (Union): このイベントがオートメーションをトリガーするために満たす必要がある追加条件（存在する場合）。
- scope (Union): イベントのスコープ。このイベントでは、artifact コレクションのみが有効なスコープとなります。

### <kbd>メソッド</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントが指定したアクションをトリガーする新しいオートメーションを定義します。
