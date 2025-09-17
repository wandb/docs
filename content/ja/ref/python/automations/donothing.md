---
title: 何もしない
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-donothing
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



意図的に何もしない オートメーション のアクションを定義します。
属性:
- action_type (Literal): トリガーするアクションの種類。
- no_op (bool): バックエンドのスキーマ要件を満たすためだけに存在するプレースホルダーのフィールドです。
    このフィールドの 値 は無視されるため、明示的に設定する必要はありません。