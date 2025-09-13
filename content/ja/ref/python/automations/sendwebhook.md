---
title: Webhook を送信
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-sendwebhook
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



Webhook リクエストを送信するオートメーション アクションを定義します。

属性:
- action_type (Literal): トリガーされるアクションの種類。
- request_payload (Optional): Webhook リクエストで送信するペイロード (テンプレート変数を含む場合があります)。

### <kbd>メソッド</kbd> `from_integration`
```python
from_integration(cls, integration: 'WebhookIntegration', *, payload: 'Optional[SerializedToJson[dict[str, Any]]]' = None) -> 'Self'
```
指定された (webhook) インテグレーションに送信する Webhook アクションを定義します。