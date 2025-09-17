---
title: ログをオフにするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-logging_turn_off
support:
- ログ
toc_hide: true
type: docs
url: /support/:filename
---

`wandb offline` コマンドは、環境変数 `WANDB_MODE=offline` を設定し、リモートの W&B サーバーへのデータ同期を行わないようにします。この設定はすべての Projects に適用され、W&B サーバーへのデータのログ記録を停止します。

警告メッセージを抑制するには、次のコードを使用します:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```