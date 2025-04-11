---
title: ログをオフにするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-logging_turn_off
support:
- logs
toc_hide: true
type: docs
url: /support/:filename
---

`wandb offline` コマンドは環境変数 `WANDB_MODE=offline` を設定し、データがリモート W&B サーバーと同期されないようにします。このアクションはすべての Projects に影響を与え、データの W&B サーバーへのログを停止します。

警告メッセージを抑制するには、以下のコードを使用します。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```