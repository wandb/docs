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

コマンド `wandb offline` は、環境変数 `WANDB_MODE=offline` を設定し、データがリモートの W&B サーバーへ同期されるのを防ぎます。この操作はすべての Projects に影響し、データが W&B サーバーにログされるのを停止します。

警告メッセージを非表示にするには、次のコードを使用してください。

```python
import logging

# wandb用のロガーを取得
logger = logging.getLogger("wandb")
# 警告レベルに設定
logger.setLevel(logging.WARNING)
```