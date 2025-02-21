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

`wandb offline` コマンド は、環境変数 `WANDB_MODE=offline` を設定し、データ がリモート の W&B サーバー に同期されないようにします。この操作はすべての プロジェクト に影響し、W&B サーバー への データ の ログ 記録を停止します。

警告メッセージ を抑制するには、次の コード を使用します。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```
