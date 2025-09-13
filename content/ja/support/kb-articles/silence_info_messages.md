---
title: W&B の 情報メッセージを抑制するには？
menu:
  support:
    identifier: ja-support-kb-articles-silence_info_messages
support:
- ノートブック
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

ノートブック で、次のような ログ メッセージを抑制するには:

```
INFO SenderThread:11484 [sender.py:finish():979]
``` 
エラーだけを表示し、INFO レベルの ログ 出力を抑制するには、ログ レベルを `logging.ERROR` に設定します。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

ログ 出力を大幅に減らすには、`WANDB_QUIET` 環境変数を `True` に設定します。ログ 出力を完全にオフにするには、`WANDB_SILENT` 環境変数を `True` に設定します。ノートブック では、`wandb.login` を実行する前に `WANDB_QUIET` または `WANDB_SILENT` を設定します:

{{< tabpane text=true langEqualsHeader=true >}}
{{% tab "ノートブック" %}}
```python
%env WANDB_SILENT=True
```
{{% /tab %}}
{{% tab "Python" %}}
```python
import os

os.environ["WANDB_SILENT"] = "True"
```
{{% /tab %}}
{{< /tabpane >}}