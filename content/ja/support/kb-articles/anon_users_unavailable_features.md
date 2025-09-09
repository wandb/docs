---
title: 匿名ユーザーが利用できない機能は何ですか？
menu:
  support:
    identifier: ja-support-kb-articles-anon_users_unavailable_features
support:
- 匿名
toc_hide: true
type: docs
url: /support/:filename
---

* **永続的なデータはありません**: 匿名 アカウント では Runs は 7 日間だけ保存されます。匿名 run の データ は、実際の アカウント に保存することで引き継げます。

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="匿名モードのインターフェース" >}}

* **Artifacts の ログはできません**: 匿名 run に Artifacts を ログ しようとすると、コマンドラインに警告が表示されます:
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **プロフィールや設定ページはありません**: UI には一部のページが含まれていません。これらは実際の アカウント でのみ有用なためです。