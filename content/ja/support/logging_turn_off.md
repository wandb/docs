---
title: How do I turn off logging?
menu:
  support:
    identifier: ja-support-logging_turn_off
tags:
- logs
toc_hide: true
type: docs
---

`wandb offline` コマンドは環境変数 `WANDB_MODE=offline` を設定し、データがリモートの W&B サーバーに同期されるのを防ぎます。この操作はすべてのプロジェクトに影響を与え、データの W&B サーバーへのログを停止します。

警告メッセージを抑制するには、次のコードを使用します。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```