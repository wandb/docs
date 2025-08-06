---
title: 匿名ユーザーには利用できない機能は何ですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 匿名
---

* **永続的なデータは保存されません**: Run は匿名アカウントで 7 日間保存されます。匿名の run データを本物のアカウントに保存することで引き継ぐことができます。

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="匿名モードのインターフェース" >}}

* **Artifact のログはできません**: 匿名 run に Artifact をログしようとすると、コマンドライン上に警告が表示されます。
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **プロフィールや設定ページがありません**: UI には特定のページが表示されません。これらは本物のアカウントにのみ必要なためです。