---
title: 自動化
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-automation
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/automations.py >}}



保存された W&B オートメーションのローカルインスタンスです。

属性:
- action (Union): このオートメーションがトリガーされたときに実行されるアクション。
- description (Optional): このオートメーションの任意の説明。
- enabled (bool): このオートメーションが有効かどうか。有効なオートメーションのみがトリガーされます。
- event (SavedEvent): このオートメーションをトリガーするイベント。
- name (str): このオートメーションの名前。
- scope (Union): トリガーとなるイベントが発生する必要があるスコープ。