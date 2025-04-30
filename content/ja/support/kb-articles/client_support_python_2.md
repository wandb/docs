---
title: W&B クライアントは Python 2 をサポートしていますか？
menu:
  support:
    identifier: ja-support-kb-articles-client_support_python_2
support:
- '{< tabpane >} {% tab header="Python" %}


  ```python

  # この例では、wandb をインポートし、それを使用してログインします。

  import wandb


  # 新しい Run を作成します

  run = wandb.init(project="my-first-project")


  # メトリクスをログ

  wandb.log({"accuracy": 0.9})


  # Run を完了します

  run.finish()

  ```


  {% /tab %} {< /tabpane >}'
toc_hide: true
type: docs
url: /ja/support/:filename
---

W&B クライアントライブラリは、バージョン 0.10 まで Python 2.7 と Python 3 の両方をサポートしていました。Python 2.7 のサポートは Python 2 の終焉に伴い、バージョン 0.11 で終了しました。Python 2.7 システムで `pip install --upgrade wandb` を実行すると、0.10.x シリーズの新しいリリースのみがインストールされます。0.10.x シリーズのサポートは重要なバグ修正とパッチのみが含まれます。Python 2.7 をサポートする 0.10.x シリーズの最後のバージョンは 0.10.33 です。