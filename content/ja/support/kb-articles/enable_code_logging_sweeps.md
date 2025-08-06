---
title: スイープでコードのログを有効にするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
---

スイープでコードのログを有効にするには、W&B Run の初期化後に `wandb.log_code()` を追加してください。W&B のプロファイル設定でコードのログが有効になっている場合でも、この操作は必要です。より高度なコードのログについては、[こちらの `wandb.log_code()` のドキュメント]({{< relref "/ref/python/sdk/classes/run#log_code" >}})をご参照ください。