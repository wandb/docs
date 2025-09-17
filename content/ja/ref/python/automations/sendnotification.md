---
title: SendNotification
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-sendnotification
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



(Slack) 通知を送信するオートメーションのアクションを定義します。

属性:
- action_type (Literal): トリガーされるアクションの種類。
- message (str): 送信される通知のメッセージ本文。
- severity (AlertSeverity): 送信される通知の重大度 (`INFO`, `WARN`, `ERROR`)。
- title (str): 送信される通知のタイトル。

### <kbd>メソッド</kbd> `from_integration`
```python
from_integration(cls, integration: 'SlackIntegration', *, title: 'str' = '', text: 'str' = '', level: 'AlertSeverity' = <AlertSeverity.INFO: 'INFO'>) -> 'Self'
```
指定した (Slack) インテグレーションに送信する通知アクションを定義します。