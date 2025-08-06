---
title: 'SendWebhook

  '
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}

オートメーションアクションを定義し、webhook リクエストを送信します。

属性:
- action_type (Literal): トリガーされるアクションの種類。
- request_payload (Optional): webhook リクエストで送信するペイロード（テンプレート変数を含む場合があります）。

### <kbd>メソッド</kbd> `from_integration`
```python
from_integration(cls, integration: 'WebhookIntegration', *, payload: 'Optional[SerializedToJson[dict[str, Any]]]' = None) -> 'Self'
```
指定した（webhook）インテグレーションに送る webhook アクションを定義します。
