---
title: 匿名ユーザーが利用できない機能は何ですか？
menu:
  support:
    identifier: ja-support-kb-articles-anon_users_unavailable_features
support:
- anonymous
toc_hide: true
type: docs
url: /support/:filename
---

* **永続的なデータはありません**: Runs は匿名アカウントで 7 日間保存されます。匿名 run データを正規のアカウントに保存することで取得できます。

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="" >}}

* **アーティファクトログはありません**: 匿名 run にアーティファクトをログしようとすると、コマンドラインに警告が表示されます。
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **プロフィールや設定ページはありません**: UI には特定のページが含まれていません。これらのページは正規のアカウントにのみ有用です。