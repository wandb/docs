---
title: What are features that are not available to anonymous users?
menu:
  support:
    identifier: ja-support-anon_users_unavailable_features
tags:
- anonymous
toc_hide: true
type: docs
---

*   **永続的なデータなし**: run は匿名アカウントに 7 日間保存されます。匿名 run のデータを実際の アカウントに保存して、データを要求してください。

    {{< img src="/images/app_ui/anon_mode_no_data.png" alt="" >}}

*   **Artifacts のログ記録なし**: 匿名 run に Artifacts を ログ記録しようとすると、 コマンドラインに警告が表示されます。
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    # wandb: 警告：匿名で記録された Artifacts は要求できず、7 日後に期限切れになります。
    ```

*   **プロフィール ページまたは 設定ページなし**: UI には、実際のアカウントでのみ役立つ特定のページは含まれていません。
