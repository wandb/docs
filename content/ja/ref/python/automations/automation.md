---
title: 自動化
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/automations.py >}}

保存された W&B オートメーション のローカルインスタンスです。

属性:
- action (Union): このオートメーションがトリガーされたときに実行されるアクション。
- description (Optional): このオートメーションの任意の説明文。
- enabled (bool): このオートメーションが有効かどうか。 有効なオートメーションのみがトリガーされます。
- event (SavedEvent): このオートメーションをトリガーするイベント。
- name (str): このオートメーションの名前。
- scope (Union): トリガーイベントが発生する必要があるスコープ。