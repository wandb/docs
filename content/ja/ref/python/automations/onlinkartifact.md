---
title: OnLinkArtifact
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-onlinkartifact
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



新しい artifact がコレクションにリンクされます。

属性:
- event_type (Literal): 説明はありません。
- filter (Union): このイベントがオートメーションをトリガーするために満たす必要がある、該当する場合の追加条件。
- scope (Union): イベントのスコープ。

### <kbd>メソッド</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
このイベントが指定のアクションをトリガーする新しいオートメーションを定義します。