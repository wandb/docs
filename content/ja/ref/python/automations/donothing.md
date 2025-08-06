---
title: 何もしない
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-donothing
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}

何もしないことを意図したオートメーションアクションを定義します。

属性:
- action_type (Literal): トリガーされるアクションの種類。
- no_op (bool): バックエンドスキーマの要件を満たすためだけに存在するプレースホルダーフィールドです。
    このフィールドを明示的に設定する必要は基本的になく、その値は無視されます。