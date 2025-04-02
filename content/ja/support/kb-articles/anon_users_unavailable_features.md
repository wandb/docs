---
title: What are features that are not available to anonymous users?
menu:
  support:
    identifier: ja-support-kb-articles-anon_users_unavailable_features
support:
- anonymous
toc_hide: true
type: docs
url: /support/:filename
---

* **永続的なデータなし**: run は匿名アカウントに 7 日間保存されます。実際の アカウントに保存することで、匿名 run の データを要求できます。

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="" >}}

* **Artifacts の ログ記録なし**: Artifacts を匿名 run に 記録しようとすると、コマンドラインに警告が表示されます。
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    # wandb: 警告 匿名でログに記録された Artifacts は要求できず、7日後に期限切れになります。
    ```

* **プロファイルまたは 設定 ページなし**: UI には、実際のアカウントでのみ役立つ特定のページは含まれていません。
