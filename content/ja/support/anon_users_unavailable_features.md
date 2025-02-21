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

* **永続的なデータなし**: Runs は匿名アカウントに7日間保存されます。匿名の run データを実際のアカウントに保存して取得しましょう。

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="" >}}

* **アーティファクトのログ不可**: 匿名 run にアーティファクトをログしようとすると、コマンドラインに警告が表示されます:
    
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **プロフィールまたは設定ページなし**: UI には特定のページが含まれていません。これらのページは実際のアカウントでのみ有用です。