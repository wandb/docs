---
title: 通知を送信
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



（Slack）通知を送信するオートメーション アクションを定義します。

属性:
- action_type (Literal): トリガーされるアクションの種類。
- message (str): 送信される通知の本文。
- severity (AlertSeverity): 送信される通知の重大度（`情報`、`WARN`、`ERROR`）。
- title (str): 送信される通知のタイトル。

### <kbd>メソッド</kbd> `from_integration`
```python
from_integration(cls, integration: 'SlackIntegration', *, title: 'str' = '', text: 'str' = '', level: 'AlertSeverity' = <AlertSeverity.INFO: 'INFO'>) -> 'Self'
```
指定した（Slack）インテグレーションに通知アクションを定義します。
