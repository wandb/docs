---
title: Does the W&B client support Python 2?
menu:
  support:
    identifier: ja-support-client_support_python_2
tags:
- python
toc_hide: true
type: docs
---

W&B クライアント ライブラリは、バージョン 0.10 まで Python 2.7 と Python 3 の両方をサポートしていました。Python 2 のサポートが終了したため、バージョン 0.11 から Python 2.7 のサポートが終了しました。Python 2.7 システムで `pip install --upgrade wandb` を実行すると、0.10.x シリーズの新しいリリースのみがインストールされます。0.10.x シリーズのサポートには、重大なバグ修正とパッチのみが含まれています。Python 2.7 をサポートする 0.10.x シリーズの最後のバージョンは 0.10.33 です。