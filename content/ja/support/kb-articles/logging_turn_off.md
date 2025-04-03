---
title: How do I turn off logging?
menu:
  support:
    identifier: ja-support-kb-articles-logging_turn_off
support:
- logs
toc_hide: true
type: docs
url: /support/:filename
---

`wandb offline` コマンドは、環境変数 `WANDB_MODE=offline` を設定し、データがリモートの W&B サーバーに同期されるのを防ぎます。この操作はすべての Projects に影響し、W&B サーバーへのデータのログ記録を停止します。

警告メッセージを抑制するには、次のコードを使用します。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```