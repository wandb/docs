---
title: W&B クライアントは Python 2 をサポートしていますか？
menu:
  support:
    identifier: ja-support-kb-articles-client_support_python_2
support:
- python
toc_hide: true
type: docs
url: /support/:filename
---

W&B クライアント ライブラリは、バージョン 0.10 まで Python 2.7 と Python 3 の両方をサポートしていました。Python 2 のサポート終了に伴い、バージョン 0.11 で Python 2.7 のサポートは終了しました。Python 2.7 のシステムで `pip install --upgrade wandb` を実行すると、0.10.x 系の新しいリリースのみがインストールされます。0.10.x 系へのサポートは、重大なバグ修正とパッチのみに限定されます。Python 2.7 をサポートする 0.10.x 系の最終バージョンは 0.10.33 です。