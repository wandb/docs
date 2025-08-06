---
title: ログをオフにするにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- ログ
---

`wandb offline` コマンドは、環境変数 `WANDB_MODE=offline` を設定し、データがリモートの W&B サーバーへ同期されないようにします。この操作はすべての Projects に影響し、W&B サーバーへのデータのログ記録を停止します。

警告メッセージを非表示にするには、次のコードを使用してください：

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```