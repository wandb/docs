---
title: OnLinkArtifact
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



新しい artifact がコレクションにリンクされました。

属性:
- event_type (Literal): 説明はありません。
- filter (Union): このイベントがオートメーションをトリガーするために満たす必要がある追加条件（該当する場合）。
- scope (Union): イベントのスコープ。

### <kbd>method</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントが指定したアクションをトリガーする新しいオートメーションを定義します。
