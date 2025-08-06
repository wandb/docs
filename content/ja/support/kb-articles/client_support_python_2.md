---
title: W&B クライアントは Python 2 をサポートしていますか？
menu:
  support:
    identifier: ja-support-kb-articles-client_support_python_2
support:
- パイソン
toc_hide: true
type: docs
url: /support/:filename
---

W&B クライアントライブラリは、バージョン 0.10 までは Python 2.7 と Python 3 の両方をサポートしていましたが、Python 2 のサポート終了に伴い、バージョン 0.11 からは Python 2.7 のサポートが終了しました。Python 2.7 のシステムで `pip install --upgrade wandb` を実行すると、0.10.x 系列の新しいリリースのみがインストールされます。0.10.x 系列のサポートは、重大なバグ修正やパッチの提供のみとなります。Python 2.7 をサポートする 0.10.x 系列の最終バージョンは 0.10.33 です。