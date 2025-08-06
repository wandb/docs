---
title: OnLinkArtifact
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-onlinkartifact
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



新しい Artifacts がコレクションにリンクされます。

属性:
- event_type (Literal): 説明はありません。
- filter (Union): このイベントがオートメーションをトリガーするために満たすべき追加の条件（あれば）。
- scope (Union): イベントのスコープ。

### <kbd>メソッド</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントが指定したアクションをトリガーする新しいオートメーションを定義します。
