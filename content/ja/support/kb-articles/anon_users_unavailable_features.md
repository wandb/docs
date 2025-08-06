---
title: 匿名ユーザーが利用できない機能には何がありますか？
menu:
  support:
    identifier: ja-support-kb-articles-anon_users_unavailable_features
support:
- 匿名
toc_hide: true
type: docs
url: /support/:filename
---

* **永続的なデータなし**: Run は匿名アカウントで 7 日間保存されます。匿名の run データを本当のアカウントに保存して引き継ぐことができます。

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="匿名モードのインターフェース" >}}

* **Artifact のロギング不可**: 匿名 run に Artifact をログしようとすると、コマンドラインに警告が表示されます。
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **プロフィールや設定ページなし**: UI にこれらのページは表示されません（本当のアカウントでのみ利用できます）。