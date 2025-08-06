---
title: SendWebhook（Webhook を送信）
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-sendwebhook
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



ウェブフックリクエストを送信するオートメーションアクションを定義します。

属性:
- action_type (リテラル): トリガーされるアクションの種類。
- request_payload (オプション): ウェブフックリクエストで送信するペイロード。テンプレート変数を含む場合があります。

### <kbd>method</kbd> `from_integration`
```python
from_integration(cls, integration: 'WebhookIntegration', *, payload: 'Optional[SerializedToJson[dict[str, Any]]]' = None) -> 'Self'
```
指定した（ウェブフック）インテグレーションに送信するウェブフックアクションを定義します。
