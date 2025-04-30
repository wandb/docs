---
title: Python コードを使って sweep を再開するにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-resume_sweep_using_python_code
support:
- スイープ
- 申し訳ありませんが、翻訳するための具体的なコンテンツを提供してください。そのコンテンツを翻訳します。
toc_hide: true
type: docs
url: /support/:filename
---

sweep を再開するには、`sweep_id` を `wandb.agent()` 関数に渡します。

```python
import wandb

sweep_id = "your_sweep_id"

def train():
    # トレーニングコードはこちら
    pass

wandb.agent(sweep_id=sweep_id, function=train)
```